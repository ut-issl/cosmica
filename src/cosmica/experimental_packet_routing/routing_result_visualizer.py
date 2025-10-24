import logging
from functools import cached_property
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.dates import DateFormatter
from matplotlib.ticker import AutoMinorLocator
from pandas import Series

from cosmica.dtos.dynamics_data import DynamicsData
from cosmica.experimental_packet_routing.dtos import (
    PacketRoutingResult,
    PacketRoutingSetting,
)
from cosmica.models.internet import Internet
from cosmica.models.satellite import ConstellationSatellite
from cosmica.visualization.equirectangular import (
    draw_countries,
    draw_lat_lon_grid,
    draw_snapshot,
)

logger = logging.getLogger(__name__)


# TODO (Takashima): 現状の描画は単一需要のみを対象としているため, 改善が必要
class RoutingResultVisualizer:
    def __init__(
        self,
        simulation_settings: PacketRoutingSetting,
        packet_routing_result: PacketRoutingResult,
    ) -> None:
        self.simulation_settings: PacketRoutingSetting = simulation_settings
        self.packet_routing_result: PacketRoutingResult = packet_routing_result

    def calculate_average_delay(
        self,
        *,
        time_from: np.datetime64 | None = None,
        time_to: np.datetime64 | None = None,
        weighted_data_size: bool = True,
    ) -> float:
        """到着したデータについて平均遅延時間を計算する."""
        calc_df = pd.DataFrame(
            {
                "generated_time": [
                    comm_data_demand.generated_time
                    for comm_data_demand in self.packet_routing_result.comm_data_demand_list
                ],
                "delay": [
                    comm_data_demand.delay for comm_data_demand in self.packet_routing_result.comm_data_demand_list
                ],
                "data_size": [
                    comm_data_demand.data_size for comm_data_demand in self.packet_routing_result.comm_data_demand_list
                ],
            },
        )

        # 時間範囲を指定してフィルタリング
        if time_from is not None:
            calc_df = calc_df[calc_df["generated_time"] >= time_from]
        if time_to is not None:
            calc_df = calc_df[calc_df["generated_time"] <= time_to]

        # 有限な遅延値のみを選択
        finite_delay_ser: Series = calc_df["delay"][np.isfinite(calc_df["delay"])]
        finite_data_size_ser: pd.Series = calc_df["data_size"][np.isfinite(calc_df["delay"])]

        if weighted_data_size:
            # データサイズで重み付けした平均遅延時間を計算
            average_delay: float = np.average(finite_delay_ser, weights=finite_data_size_ser)
        else:
            # 単純な平均遅延時間を計算
            average_delay = finite_delay_ser.mean()

        logger.info(f"Average delay: {average_delay} s")

        return average_delay

    def calculate_average_increased_delay(
        self,
        time_baseline: np.datetime64,
        *,
        time_from: np.datetime64 | None = None,
        time_to: np.datetime64 | None = None,
        weighted_data_size: bool = True,
    ) -> float:
        """到着したデータについて平均遅延時間の、ある時刻の遅延時間に対する増加量を計算する."""
        calc_df = pd.DataFrame(
            {
                "generated_time": [
                    comm_data_demand.generated_time
                    for comm_data_demand in self.packet_routing_result.comm_data_demand_list
                ],
                "delay": [
                    comm_data_demand.delay for comm_data_demand in self.packet_routing_result.comm_data_demand_list
                ],
                "data_size": [
                    comm_data_demand.data_size for comm_data_demand in self.packet_routing_result.comm_data_demand_list
                ],
            },
        )

        # 時間範囲を指定してフィルタリング
        if time_from is not None:
            calc_df = calc_df[calc_df["generated_time"] >= time_from]
        if time_to is not None:
            calc_df = calc_df[calc_df["generated_time"] <= time_to]

        # 有限な遅延値のみを選択
        finite_delay_ser: Series = calc_df["delay"][np.isfinite(calc_df["delay"])]
        finite_data_size_ser: pd.Series = calc_df["data_size"][np.isfinite(calc_df["delay"])]

        if weighted_data_size:
            # データサイズで重み付けした平均遅延時間を計算
            average_delay: float = np.average(
                finite_delay_ser,
                weights=finite_data_size_ser,
            )
        else:
            # 単純な平均遅延時間を計算
            average_delay = finite_delay_ser.mean()

        # 指定した時刻の遅延時間からの差分を計算
        closest_idx = (calc_df["generated_time"] - pd.to_datetime(time_baseline)).abs().idxmin()
        average_increased_delay = average_delay - pd.to_numeric(
            calc_df.loc[closest_idx, "delay"],
        )

        logger.info(f"Average increased delay: {average_increased_delay} s")

        return average_delay

    def calculate_average_packet_loss_rate(
        self,
        *,
        time_from: np.datetime64 | None = None,
        time_to: np.datetime64 | None = None,
        weighted_data_size: bool = True,
    ) -> float:
        """生成データの平均パケットロス率を計算する."""
        calc_df = pd.DataFrame(
            {
                "generated_time": [
                    comm_data_demand.generated_time
                    for comm_data_demand in self.packet_routing_result.comm_data_demand_list
                ],
                "reach_dst": [
                    comm_data_demand.reach_dst for comm_data_demand in self.packet_routing_result.comm_data_demand_list
                ],
                "data_size": [
                    comm_data_demand.data_size for comm_data_demand in self.packet_routing_result.comm_data_demand_list
                ],
            },
        )

        # 時間範囲を指定してフィルタリング
        if time_from is not None:
            calc_df = calc_df[calc_df["generated_time"] >= time_from]
        if time_to is not None:
            calc_df = calc_df[calc_df["generated_time"] <= time_to]

        if weighted_data_size:
            # データサイズで重み付けした平均パケットロス率を計算
            average_packet_reach_rate: float = np.average(calc_df["reach_dst"], weights=calc_df["data_size"])
        else:
            average_packet_reach_rate = calc_df["reach_dst"].mean()

        average_packet_loss_rate = 1 - average_packet_reach_rate
        logger.info(f"Average packet loss rate: {average_packet_loss_rate}")

        return average_packet_loss_rate

    def plot_delay(
        self,
        save_path: Path,
        *,
        with_title: bool = False,
        title_fontsize: int = 16,
        label_fontsize: int = 14,
        legend_fontsize: int = 12,
        tick_label_fontsize: int = 12,
        legend_loc: str = "best",
        xlim: tuple | None = None,
        ylim: tuple | None = None,
        dpi: int = 300,
        with_grid: bool = False,
        use_time_from_epoch: bool = False,
        replace_inf: bool = False,
        # link_failure_timing=None,
    ) -> None:
        ## Convert comm_data_demand_list into a pandas DataFrame
        # 到達したデータ, もしくはパケットロスしたデータのみプロットする
        plot_df = pd.DataFrame(
            {
                "generated_time": [
                    comm_data_demand.generated_time
                    for comm_data_demand in self.packet_routing_result.comm_data_demand_list
                ],
                "delay": [
                    comm_data_demand.delay for comm_data_demand in self.packet_routing_result.comm_data_demand_list
                ],
            },
        )
        plot_df = plot_df.sort_values("generated_time")

        if use_time_from_epoch:
            x_data = (plot_df["generated_time"] - self.time[0]) / np.timedelta64(1, "s")
        else:
            x_data = plot_df["generated_time"]

        max_delay = plot_df["delay"][plot_df["delay"] != np.inf].max()

        if replace_inf:
            # np.infだとプロットされないので、適当な大きい値に置換
            plot_df["delay"] = plot_df["delay"].replace(np.inf, 1e5)

        ## プロット
        fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)

        ax.plot(x_data, plot_df["delay"], label="Delay", color="blue")

        # Customizing title and labels font size
        if with_title:
            ax.set_title("Delay", fontsize=title_fontsize)
        ax.set_ylabel("Average delay time [s]", fontsize=label_fontsize)
        if use_time_from_epoch:
            ax.set_xlabel("Time from epoch[s]", fontsize=label_fontsize)
        else:
            ax.set_xlabel("Time", fontsize=label_fontsize)
            # Formatting the x-axis with date and time, and rotating the text
            ax.xaxis.set_major_formatter(DateFormatter("%Y/%m/%d %H:%M:%S"))
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        # Customizing tick label font size
        ax.tick_params(axis="both", labelsize=tick_label_fontsize)

        # 軸の範囲設定
        if use_time_from_epoch:
            ax.set_xlim(xlim if xlim else (self.time_from_epoch[0], self.time_from_epoch[-1]))
        else:
            ax.set_xlim(xlim if xlim else (self.time[0], self.time[-1]))

        if max_delay == np.inf or np.isnan(max_delay):
            ax.set_ylim(0, None)
        else:
            ax.set_ylim(ylim if ylim else (0, max_delay * 1.1))

        if with_grid:
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
            ax.grid(which="major", color="#CCCCCC", linestyle="--")
            ax.grid(which="minor", color="#CCCCCC", linestyle=":")

        # 凡例の設定
        ax.legend(fontsize=legend_fontsize, loc=legend_loc)

        # 保存
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved visualization: {save_path}")

    def plot_arrival_data_rate(
        self,
        save_path: Path,
        *,
        with_title: bool = False,
        title_fontsize: int = 16,
        label_fontsize: int = 14,
        legend_fontsize: int = 12,
        tick_label_fontsize: int = 12,
        legend_loc: str = "best",
        xlim: tuple | None = None,
        ylim: tuple | None = None,
        dpi: int = 300,
        with_grid: bool = False,
        use_time_from_epoch: bool = False,
        # link_failure_timing=None,
    ) -> None:
        # --- 1. 必要データを整形 ---------------------------------------------------
        plot_df = pd.DataFrame(
            {
                "generated_time": [
                    comm_data_demand.generated_time
                    for comm_data_demand in self.packet_routing_result.comm_data_demand_list
                ],
                "delay": [
                    comm_data_demand.delay for comm_data_demand in self.packet_routing_result.comm_data_demand_list
                ],
                "data_size": [
                    comm_data_demand.data_size for comm_data_demand in self.packet_routing_result.comm_data_demand_list
                ],
                "reach_dst": [
                    comm_data_demand.reach_dst for comm_data_demand in self.packet_routing_result.comm_data_demand_list
                ],
            },
        )
        plot_df = plot_df[plot_df["reach_dst"]]
        plot_df["arrival_time"] = (plot_df["generated_time"] + pd.to_timedelta(plot_df["delay"], unit="s")).astype(
            "datetime64[ms]",
        )

        # --- 2. タイムステップ情報 --------------------------------------------------
        time_index = pd.Series(self.time).astype("datetime64[ms]")
        dt_seconds = np.asarray(self.time_step_array, dtype=float)

        # --- 3. 各 arrival を直前の time_index に割り当て ---------------------------
        # np.searchsorted で挿入位置を取得 → 1 つ前のインデックスが「属するステップ」
        idx = np.searchsorted(time_index.to_numpy(), plot_df["arrival_time"].to_numpy(), side="right") - 1
        valid = idx >= 0  # 負インデックスは epoch 以前なので除外

        # --- 4. ステップ毎にバイト数を加算 -----------------------------------------
        bytes_per_step = np.zeros(len(time_index), dtype=float)
        np.add.at(bytes_per_step, idx[valid], plot_df["data_size"].to_numpy()[valid])

        arrival_rate = bytes_per_step / dt_seconds

        # --- 5. 可視化 -------------------------------------------------------------
        x_data = (time_index - time_index.iloc[0]).dt.total_seconds() if use_time_from_epoch else time_index

        fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
        ax.plot(x_data, arrival_rate, label="Arrival data rate", lw=1)

        if with_title:
            ax.set_title("Arrival Data Rate", fontsize=title_fontsize)
        ax.set_ylabel("Arrival data rate [bit/s]", fontsize=label_fontsize)
        if use_time_from_epoch:
            ax.set_xlabel("Time from Epoch [s]", fontsize=label_fontsize)
        else:
            ax.set_xlabel("Time", fontsize=label_fontsize)
            ax.xaxis.set_major_formatter(DateFormatter("%Y/%m/%d %H:%M:%S"))
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        ax.tick_params(axis="both", labelsize=tick_label_fontsize)
        ax.set_xlim(xlim if xlim else (x_data.iloc[0], x_data.iloc[-1]))
        ax.set_ylim(ylim if ylim else (0, None))

        if with_grid:
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
            ax.grid(which="major", linestyle="--", alpha=0.6)
            ax.grid(which="minor", linestyle=":", alpha=0.4)

        ax.legend(fontsize=legend_fontsize, loc=legend_loc)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved visualization: {save_path}")

    def plot_accumulated_arrival_data_size(
        self,
        save_path: Path,
        *,
        with_title: bool = False,
        title_fontsize: int = 16,
        label_fontsize: int = 14,
        legend_fontsize: int = 12,
        tick_label_fontsize: int = 12,
        legend_loc: str = "best",
        xlim: tuple | None = None,
        ylim: tuple | None = None,
        dpi: int = 300,
        with_grid: bool = False,
        use_time_from_epoch: bool = False,
        # link_failure_timing=None,
    ) -> None:
        ## Convert comm_data_demand_list into a pandas DataFrame
        # 到達したデータ, もしくはパケットロスしたデータのみプロットする
        plot_df = pd.DataFrame(
            {
                "generated_time": [
                    comm_data_demand.generated_time
                    for comm_data_demand in self.packet_routing_result.comm_data_demand_list
                ],
                "delay": [
                    comm_data_demand.delay for comm_data_demand in self.packet_routing_result.comm_data_demand_list
                ],
                "data_size": [
                    comm_data_demand.data_size for comm_data_demand in self.packet_routing_result.comm_data_demand_list
                ],
                "reach_dst": [
                    comm_data_demand.reach_dst for comm_data_demand in self.packet_routing_result.comm_data_demand_list
                ],
            },
        )
        plot_df = plot_df[plot_df["reach_dst"]]

        plot_df["arrival_time"] = plot_df["generated_time"] + pd.to_timedelta(plot_df["delay"], unit="s")
        plot_df = plot_df.sort_values("arrival_time")

        plot_df["accumulated_arrival_data_size"] = plot_df.groupby("reach_dst")["data_size"].cumsum()

        if use_time_from_epoch:
            x_data = (plot_df["arrival_time"] - self.time[0]) / np.timedelta64(1, "s")
        else:
            x_data = plot_df["arrival_time"]

        ## プロット
        fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)

        ax.plot(
            x_data,
            plot_df["accumulated_arrival_data_size"],
            label="Accumulated arrival data size",
            color="blue",
        )

        # Customizing title and labels font size
        if with_title:
            ax.set_title("Accumulated Arrival Data Size", fontsize=title_fontsize)
        ax.set_ylabel("Accumulated arrival data size [bit]", fontsize=label_fontsize)
        if use_time_from_epoch:
            ax.set_xlabel("Time from Epoch [s]", fontsize=label_fontsize)
        else:
            ax.set_xlabel("Time", fontsize=label_fontsize)
            # Formatting the x-axis with date and time, and rotating the text
            ax.xaxis.set_major_formatter(DateFormatter("%Y/%m/%d %H:%M:%S"))
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        # Customizing tick label font size
        ax.tick_params(axis="both", labelsize=tick_label_fontsize)

        # 軸の範囲設定
        if use_time_from_epoch:
            ax.set_xlim(xlim if xlim else (self.time_from_epoch[0], self.time_from_epoch[-1]))
        else:
            ax.set_xlim(xlim if xlim else (self.time[0], self.time[-1]))
        ax.set_ylim(ylim if ylim else (0, None))

        if with_grid:
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
            ax.grid(which="major", color="#CCCCCC", linestyle="--")
            ax.grid(which="minor", color="#CCCCCC", linestyle=":")

        # 凡例の設定
        ax.legend(fontsize=legend_fontsize, loc=legend_loc)

        # 保存
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved visualization: {save_path}")

    def plot_graph_at_certain_time(
        self,
        time: np.datetime64,
        dynamics_data: DynamicsData[ConstellationSatellite],
        save_path: Path,
        *,
        with_demand_data: bool = True,
        with_lsa_data: bool = False,
        dpi: int = 100,
    ) -> None:
        """指定した時間のグラフを描画する."""
        time_indices = np.where(self.time == time)[0]
        if len(time_indices) == 0:
            msg = f"Time {time} not found in self.time array."
            raise ValueError(msg)
        time_index = time_indices[0]

        edges_with_demand_data = self._get_edges_with_demand_data(time_index) if with_demand_data else set()

        edges_with_lsa_data = self._get_edges_with_lsa_data(time_index) if with_lsa_data else set()
        # routing_table_arrows = get_routing_table_arrows(time_idx)

        fig, ax = plt.subplots(figsize=(15, 8))

        title = f"Time: {np.datetime_as_string(time, unit='ms').split('T')[1]}"
        ax.set_title(title)

        draw_lat_lon_grid(ax=ax)
        draw_countries(ax=ax)
        draw_snapshot(
            graph=self.packet_routing_result.all_graphs_after_simulation[time_index],
            dynamics_data=dynamics_data[time_index],
            ax=ax,
            with_labels=False,
            focus_edges_list=[edges_with_demand_data, edges_with_lsa_data],
            focus_edges_label_list=["Demand data", "LSA data"],
        )

        ax.legend(loc="lower left")

        # 保存
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        logger.info(f"Saved visualization: {save_path}")

    def plot_graph_over_time(
        self,
        time_index_from: int,
        time_index_to: int,
        time_index_step: int,
        dynamics_data: DynamicsData[ConstellationSatellite],
        save_path: Path,
        *,
        with_demand_data: bool = True,
        with_lsa_data: bool = False,
        dpi: int = 100,
    ) -> None:
        """時間経過に伴うグラフの変化を描画する."""
        # TODO(Takashima): time_index でなく timeを引数に取れるようにする
        n_fig = (time_index_to - time_index_from) // time_index_step
        ncols = 3
        nrows = (n_fig + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(15 * ncols, 8 * nrows),
        )
        fig.subplots_adjust(wspace=0.1, hspace=0.1)

        for idx, time_index in enumerate(range(time_index_from, time_index_to, time_index_step)):
            row, col = divmod(idx, ncols)

            edges_with_demand_data = self._get_edges_with_demand_data(time_index) if with_demand_data else set()

            edges_with_lsa_data = self._get_edges_with_lsa_data(time_index) if with_lsa_data else set()

            title = f"Time: {np.datetime_as_string(self.time[time_index], unit='ms').split('T')[1]}"
            axes[row, col].set_title(title)

            draw_lat_lon_grid(ax=axes[row, col])
            draw_countries(ax=axes[row, col])
            draw_snapshot(
                graph=self.packet_routing_result.all_graphs_after_simulation[time_index],
                dynamics_data=dynamics_data[time_index],
                ax=axes[row, col],
                with_labels=False,
                focus_edges_list=[edges_with_demand_data, edges_with_lsa_data],
                focus_edges_label_list=["Demand data", "LSA data"],
            )

        # 保存
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        logger.info(f"Saved visualization: {save_path}")

    def plot_graph_animation(
        self,
        time_index_from: int,
        time_index_to: int,
        time_index_step: int,
        dynamics_data: DynamicsData[ConstellationSatellite],
        save_path: Path,
        *,
        with_demand_data: bool = True,
        with_lsa_data: bool = False,
        dpi: int = 100,
    ) -> None:
        """時間経過に伴うグラフの変化をアニメーションで描画する."""
        # TODO(Takashima): time_index でなく timeを引数に取れるようにする
        time_index_for_plot = range(time_index_from, time_index_to, time_index_step)

        fig, ax = plt.subplots(figsize=(15, 8))

        def update(frame: int):  # noqa: ANN202
            ax.clear()
            time_index = time_index_for_plot[frame]

            edges_with_demand_data = self._get_edges_with_demand_data(time_index) if with_demand_data else set()

            edges_with_lsa_data = self._get_edges_with_lsa_data(time_index) if with_lsa_data else set()

            title = f"Time: {np.datetime_as_string(self.time[time_index], unit='ms').split('T')[1]}"
            ax.set_title(title)

            draw_lat_lon_grid(ax=ax)
            draw_countries(ax=ax)
            draw_snapshot(
                graph=self.packet_routing_result.all_graphs_after_simulation[time_index],
                dynamics_data=dynamics_data[time_index],
                ax=ax,
                with_labels=False,
                focus_edges_list=[edges_with_demand_data, edges_with_lsa_data],
                focus_edges_label_list=["Demand data", "LSA data"],
            )
            ax.legend(loc="lower left")

        ani = FuncAnimation(fig, update, frames=len(time_index_for_plot), interval=100)
        ani.save(
            save_path,
            dpi=dpi,
        )

    def _get_edges_with_demand_data(
        self,
        time_idx: int,
    ) -> set:
        edges_with_data: set = {
            edge
            for edge in self.packet_routing_result.all_graphs_after_simulation[time_idx].edges
            if self.packet_routing_result.all_graphs_after_simulation[time_idx].edges[edge][
                "bandwidth_usage_for_demand_data"
            ]
            > 0
        }
        return {edge for edge in edges_with_data if all(not isinstance(node, Internet) for node in edge)}

    def _get_edges_with_lsa_data(
        self,
        time_idx: int,
    ) -> set:
        edges_with_data: set = {
            edge
            for edge in self.packet_routing_result.all_graphs_after_simulation[time_idx].edges
            if self.packet_routing_result.all_graphs_after_simulation[time_idx].edges[edge][
                "bandwidth_usage_for_lsa_data"
            ]
            > 0
        }
        return {edge for edge in edges_with_data if all(not isinstance(node, Internet) for node in edge)}

    # def plot_bandwidth_on_edges(
    #     self,
    #     save_path: Path,
    #     *,
    #     title_fontsize: int = 16,
    #     label_fontsize: int = 14,
    #     legend_fontsize: int = 12,
    #     tick_label_fontsize: int = 12,
    #     legend_loc: str = "best",
    #     xlim: tuple | None = None,
    #     ylim: tuple | None = None,
    #     dpi: int = 300,
    #     with_grid: bool = False,
    #     use_time_from_epoch: bool = False,
    #     save_name: str = "bandwidth_on_edges",
    #     # link_failure_timing=None,
    # ) -> None:
    #     def get_bandwidth_usage_on_edge(
    #         graphs: list[nx.Graph],
    #         edge_name: tuple[Any, Any],
    #         time_array: npt.NDArray,
    #     ) -> ndarray[float]:
    #         """指定したエッジの帯域幅使用量を取得する関数."""
    #         bandwidth_usage = np.zeros(len(time_array))
    #         for time_idx in range(len(time_array)):
    #             if graphs[time_idx].has_edge(*edge_name):
    #                 bandwidth_usage[time_idx] = graphs[time_idx].edges[edge_name]["bandwidth_usage_for_demand"]
    #         return bandwidth_usage

    #     ## Convert comm_data_demand_list into a pandas DataFrame
    #     # 到達したデータ, もしくはパケットロスしたデータのみプロットする
    #     plot_df = pd.DataFrame(
    #         {
    #             "generated_time":
    # [comm_data_demand.generated_time for comm_data_demand in self.comm_data_demand_list],
    #             "delay": [comm_data_demand.delay for comm_data_demand in self.comm_data_demand_list],
    #             "data_size": [comm_data_demand.data_size for comm_data_demand in self.comm_data_demand_list],
    #             "reach_dst": [comm_data_demand.reach_dst for comm_data_demand in self.comm_data_demand_list],
    #         },
    #     )
    #     plot_df["arrival_time"] = plot_df["generated_time"] + pd.to_timedelta(plot_df["delay"], unit="s")
    #     plot_df: DataFrame = plot_df.sort_values("arrival_time")

    #     plot_df["accumulated_arrival_data_size"] = plot_df.groupby("reach_dst")["data_size"].cumsum()

    #     if use_time_from_epoch:
    #         x_data = (plot_df["arrival_time"] - self.time[0]) / np.timedelta64(1, "s")
    #     else:
    #         x_data = plot_df["arrival_time"]

    #     # 各エッジに対して帯域幅使用量を取得
    #     for graph_idx, _graph in enumerate(self.all_graphs_after_simulation)):
    #         for edge in _graph.edges:
    #             bandwidth_usage_on_edge =
    # get_bandwidth_usage_on_edge(self.all_graphs_after_simulation, edge, time_array)
    #             if np.any(bandwidth_usage_on_edge >= 0.8e9):  # 帯域幅使用量が閾値を超えた場合のみプロット
    #                 edge_label = ", ".join([key for key, value in nodes_dict.items() if value in (edge[0], edge[1])])
    #                 ax.plot(time_array_from_start_time, bandwidth_usage_on_edge, label=edge_label)

    #     ## プロット
    #     fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)

    #     ax.plot(x_data, plot_df["accumulated_arrival_data_size"], label="Accumulated arrival data size", color="blue")

    #     # Customizing title and labels font size
    #     ax.set_title("Accumulated Arrival Data Size", fontsize=title_fontsize)
    #     ax.set_ylabel("Accumulated arrival data size [bit]", fontsize=label_fontsize)
    #     if use_time_from_epoch:
    #         ax.set_xlabel("Time from Epoch [s]", fontsize=label_fontsize)
    #     else:
    #         ax.set_xlabel("Time", fontsize=label_fontsize)
    #         # Formatting the x-axis with date and time, and rotating the text
    #         ax.xaxis.set_major_formatter(DateFormatter("%Y/%m/%d %H:%M:%S"))
    #         plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    #     # Customizing tick label font size
    #     ax.tick_params(axis="both", labelsize=tick_label_fontsize)

    #     # 軸の範囲設定
    #     if use_time_from_epoch:
    #         ax.set_xlim(xlim if xlim else (self.time_from_epoch[0], self.time_from_epoch[-1]))
    #     else:
    #         ax.set_xlim(xlim if xlim else (self.time[0], self.time[-1]))
    #     ax.set_ylim(ylim if ylim else (0, None))

    #     if with_grid:
    #         ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    #         ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    #         ax.grid(which="major", color="#CCCCCC", linestyle="--")
    #         ax.grid(which="minor", color="#CCCCCC", linestyle=":")

    #     # 凡例の設定
    #     ax.legend(fontsize=legend_fontsize, loc=legend_loc)

    #     # 保存
    #     save_path: Path = save_path / (save_name + ".png")
    #     fig.savefig(save_path, bbox_inches="tight")
    #     logging.info(f"Saved visualization: {save_path}")

    @cached_property
    def time(self) -> npt.NDArray:
        return self.simulation_settings.time

    @cached_property
    def time_from_epoch(self) -> npt.NDArray:
        return (self.simulation_settings.time - self.simulation_settings.time[0]) / np.timedelta64(1, "s")

    @cached_property
    def time_step_array(self) -> npt.NDArray:
        time_from_epoch_diff = np.diff(self.time_from_epoch)
        return np.append(time_from_epoch_diff, time_from_epoch_diff[-1])
