from time import sleep
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

def render_status(finished, current, extra=""):
    table = Table.grid(padding=(0,1))
    for i, label in enumerate(["Download data", "Process records", "Train model", "Export results"]):
        mark = "[green]✓[/]" if i < finished else ("[cyan]…[/]" if i == current else " ")
        table.add_row(f"[{mark}] {label}")
    return Panel.fit(table, title="Pipeline", subtitle=extra)

def main():
    with Live(render_status(0, 0, "starting"), refresh_per_second=20, transient=False) as live:
        # Step 1
        for _ in range(30):
            live.update(render_status(0, 0, "fetching"))
            sleep(0.05)
        live.update(render_status(1, 1, "downloaded"))
        sleep(0.2)

        # Step 2 with embedded progress
        progress = Progress(
            TextColumn("Process records"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        )
        task_id = progress.add_task("proc", total=100)
        for i in range(100):
            progress.update(task_id, advance=1)
            live.update(Panel.fit(progress, title="Pipeline"))
            sleep(0.02)
        live.update(render_status(2, 2, "processed"))
        sleep(0.2)

        # Step 3
        for _ in range(40):
            live.update(render_status(2, 2, "training"))
            sleep(0.05)
        live.update(render_status(3, 3, "trained"))
        sleep(0.2)

        # Step 4
        for _ in range(10):
            live.update(render_status(3, 3, "exporting"))
            sleep(0.05)
        live.update(render_status(4, 4, "done ✅"))

if __name__ == "__main__":
    main()
