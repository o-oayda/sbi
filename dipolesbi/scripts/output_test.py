# pip install rich
import itertools
import math
import time
from collections import deque
from threading import Lock
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from contextlib import contextmanager


class PipelineUI:
    def __init__(self, tasks, steps_with_progress=None, title="Pipeline", log_lines=8):
        self.tasks = list(tasks)
        self.title = title
        self.finished = 0
        self.current = 0
        self.subtitle = "starting"
        self.steps_with_progress = set(steps_with_progress or [])
        self._progress = None
        self._task_id = None
        self._logs = deque(maxlen=log_lines)
        self._live = None
        self._lock = Lock()

    @contextmanager
    def session(self, refresh_per_second=20, transient=False):
        from rich.live import Live
        with Live(self.render(), refresh_per_second=refresh_per_second, transient=transient) as live:
            self._live = live
            try:
                yield self
            finally:
                self._live = None

    def _refresh(self):
        # safe to call from anywhere in same thread; guard just in case
        if self._live is not None:
            self._live.update(self.render())

    # ---------- public API ----------
    def start_step(self, idx, subtitle=None, total=None):
        self.current = idx
        if subtitle is not None:
            self.subtitle = subtitle

        if idx in self.steps_with_progress:
            if self._progress is None:
                self._progress = Progress(
                    TextColumn("{task.description}"),
                    BarColumn(),
                    TextColumn("{task.completed}/{task.total}"),
                    TextColumn("{task.percentage:>3.0f}%"),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    transient=False,
                    expand=True,
                )
            self._progress.stop()
            self._progress.tasks.clear()
            self._task_id = self._progress.add_task(self.tasks[idx], total=(total or 100), start=False)
        else:
            self._task_id = None
        self._refresh()

    def advance_progress(self, advance=1, total=None):
        if self._progress is not None and self._task_id is not None:
            if total is not None and math.isfinite(total):
                self._progress.update(self._task_id, total=total)
            if not self._progress.tasks[self._task_id].started:
                self._progress.start_task(self._task_id)
            self._progress.update(self._task_id, advance=advance)
        self._refresh()

    def set_subtitle(self, text):
        self.subtitle = text
        self._refresh()

    def log(self, message, style=""):
        # style examples: "dim", "yellow", "red", "green", "italic cyan"
        t = Text.from_markup(message) if '[' in message and ']' in message else Text(message)
        if style:
            t.stylize(style)
        self._logs.append(t)
        self._refresh()

    def finish_step(self, subtitle=None):
        self.finished = max(self.finished, self.current + 1)
        if subtitle is not None:
            self.subtitle = subtitle
        if self._progress is not None and self._task_id is not None:
            remaining = (self._progress.tasks[self._task_id].total
                         - self._progress.tasks[self._task_id].completed)
            if remaining > 0:
                self._progress.update(self._task_id, advance=remaining)
        self._refresh()

    def is_done(self):
        return self.finished >= len(self.tasks)

    # ---------- rendering ----------
    def _render_checklist(self):
        table = Table.grid(padding=(0, 1))
        for i, label in enumerate(self.tasks):
            if i < self.finished:
                mark = "[green]✓[/]"
                style = "green"
            elif i == self.current:
                mark = "[cyan]…[/]"
                style = "cyan"
            else:
                mark = " "
                style = "dim"
            table.add_row(f"[{mark}] [bold {style}]{label}[/]")
        return table

    def _render_logs(self):
        if not self._logs:
            return Panel("No logs yet.", title="Logs", border_style="dim")
        # Newest at bottom; show as a simple grid
        log_table = Table.grid(padding=(0, 1))
        for line in self._logs:
            log_table.add_row(line)
        return Panel(log_table, title="Logs", border_style="grey50")

    def render(self):
        body = [self._render_checklist()]
        if self._task_id is not None and self.current in self.steps_with_progress:
            body.append(self._progress)
        body.append(self._render_logs())
        return Panel(
            Group(*body),
            title=self.title,
            subtitle=self.subtitle,
            border_style="white",
        )

# ------------------- Demo -------------------

def demo():
    tasks = ["Download data", "Process records", "Train model", "Export results"]
    ui = PipelineUI(tasks, steps_with_progress={1}, title="Pipeline", log_lines=6)

    spinner = itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")

    with Live(ui.render(), refresh_per_second=20, transient=False) as live:
        # Step 1
        ui.start_step(0, "fetching")
        ui.log("Starting download…", "dim")
        for i in range(40):
            if i % 10 == 0:
                ui.log(f"Chunk {i//10 + 1}/4 received")
            ui.set_subtitle(f"fetching {next(spinner)}")
            live.update(ui.render()); time.sleep(0.05)
        ui.finish_step("downloaded")
        ui.log("[green]Download complete[/green]")
        live.update(ui.render()); time.sleep(0.2)

        # Step 2: inline progress
        ui.start_step(1, "processing", total=100)
        ui.log("Parsing 10k records…")
        for i in range(100):
            if i % 20 == 0 and i > 0:
                ui.log(f"Processed {i}%")
            ui.advance_progress(1)
            live.update(ui.render()); time.sleep(0.02)
        ui.finish_step("processed")
        ui.log("[green]Processing done[/green]")
        live.update(ui.render()); time.sleep(0.2)

        # Step 3
        ui.start_step(2, "training")
        ui.log("Training epoch 1/3")
        for i in range(60):
            if i in (20, 40):
                epoch = 1 if i == 20 else 2
                ui.log(f"Training epoch {epoch+1}/3")
            ui.set_subtitle(f"training {next(spinner)}")
            live.update(ui.render()); time.sleep(0.05)
        ui.finish_step("trained")
        ui.log("[green]Model converged[/green]")
        live.update(ui.render()); time.sleep(0.2)

        # Step 4
        ui.start_step(3, "exporting")
        ui.log("Writing artifacts…")
        for _ in range(12):
            live.update(ui.render()); time.sleep(0.05)
        ui.finish_step("done ✅")
        ui.log("[bold green]:tada: Pipeline completed[/bold green]")
        live.update(ui.render()); time.sleep(0.2)

if __name__ == "__main__":
    demo()
