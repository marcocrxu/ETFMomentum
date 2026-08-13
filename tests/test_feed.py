"""数据层测试 (头部回补 & 缓存逻辑, 需要 tushare token)"""
import os, tempfile, shutil
import numpy as np
import pandas as pd
import pytest

from data.feed import TushareFeed


@pytest.fixture
def empty_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _token_or_skip():
    """回测缓存中已有 token 文件, 通过 config_local 或环境变量取"""
    try:
        from web.config_local import TUSHARE_TOKEN
        return TUSHARE_TOKEN
    except ImportError:
        tok = os.environ.get("TUSHARE_TOKEN", "")
        if not tok:
            pytest.skip("无 TUSHARE_TOKEN")
        return tok


class TestHeadBackfill:
    """Bug A 回归: 头部回补使早于 cache_min 的请求也有数据"""

    def test_head_backfill_triggers_and_extends_range(self, empty_dir):
        token = _token_or_skip()
        feed = TushareFeed(token, cache_dir=empty_dir)

        # 1) 先建一个浅缓存: 只拉最近 200 个自然日 ≈ 140 交易日
        recent_end = "20260728"
        recent_start_dt = pd.Timestamp(recent_end) - pd.Timedelta(days=200)
        recent_start = recent_start_dt.strftime("%Y%m%d")
        dfs = feed.get_dataframe(["000001.SH"], recent_start, recent_end)
        assert "000001.SH" in dfs, "基础拉取失败"
        cache_min1 = dfs["000001.SH"]["trade_date"].min()

        # 2) 请求一个明显更早的区间 (需要跨越同一实例以便测试逻辑)
        # _load_or_fetch 在同一个 feed 实例中缓存已是刚才拉的
        early_end_dt = pd.Timestamp(recent_start)
        early_start = (pd.Timestamp(recent_start) - pd.Timedelta(days=365)).strftime("%Y%m%d")
        dfs2 = feed.get_dataframe(["000001.SH"], early_start, recent_start)
        assert "000001.SH" in dfs2, "头部回补后拉取失败"
        cache_min2 = dfs2["000001.SH"]["trade_date"].min()

        assert cache_min2 < cache_min1, \
            f"头部回补后 min={cache_min2.date()} 应早于回补前 min={cache_min1.date()}"

    def test_head_backfill_persisted_and_reload_consistent(self, empty_dir):
        token = _token_or_skip()
        feed = TushareFeed(token, cache_dir=empty_dir)

        end = "20260728"
        mid = (pd.Timestamp(end) - pd.Timedelta(days=100)).strftime("%Y%m%d")
        early = (pd.Timestamp(end) - pd.Timedelta(days=400)).strftime("%Y%m%d")

        # 先拉 mid-end
        feed.get_dataframe(["000001.SH"], mid, end)
        # 再拉 early-mid (触发头部回补并写入磁盘)
        dfs = feed.get_dataframe(["000001.SH"], early, end)
        assert "000001.SH" in dfs

        # 新实例读同一缓存: 应有完整范围
        feed2 = TushareFeed(token, cache_dir=empty_dir)
        dfs2 = feed2.get_dataframe(["000001.SH"], early, end)
        assert "000001.SH" in dfs2
        assert dfs2["000001.SH"]["trade_date"].min() <= pd.Timestamp(early), \
            "重载后缓存头部不完整"


class TestCacheIncrement:
    """向后增量回补 (已有逻辑, 确保未被头部回补破坏)"""

    def test_forward_increment_still_works(self, empty_dir):
        token = _token_or_skip()
        feed = TushareFeed(token, cache_dir=empty_dir)

        # 老数据
        old_end = "20260701"
        old_start = (pd.Timestamp(old_end) - pd.Timedelta(days=30)).strftime("%Y%m%d")
        dfs = feed.get_dataframe(["000001.SH"], old_start, old_end)
        assert "000001.SH" in dfs
        n1 = len(dfs["000001.SH"])

        # 延长到更新日期 (触发尾部增量)
        new_end = "20260728"
        dfs2 = feed.get_dataframe(["000001.SH"], old_start, new_end)
        assert "000001.SH" in dfs2
        n2 = len(dfs2["000001.SH"])
        assert n2 >= n1, f"尾部增量应不减少行数: {n1} → {n2}"
