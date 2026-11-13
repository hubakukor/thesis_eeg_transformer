"""Local Streamlit interface: python -m streamlit run dashboard.py."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from experiments import (ACTIVE, CLASSES, ROOT, RUNS_DIR, ExperimentConfig, cancel_run,
                         get_epochs, get_run, list_runs, recover_interrupted_runs, start_run)


st.set_page_config(page_title="EEG Experiment Lab", page_icon="🧠", layout="wide")
st.title("EEG Experiment Lab")
st.caption("Configure a Transformer run, follow training, and compare saved experiments.")


def comparison_frame(runs):
    rows = []
    for run in runs:
        config = json.loads(run["config_json"])
        result = json.loads(run["result_json"]) if run["result_json"] else {}
        rows.append({"run_id": run["id"], "name": config["name"], "dataset": config["dataset"],
                     "status": run["status"], "created_utc": run["created_at"],
                     "epochs": config["epochs"], "batch_size": config["batch_size"],
                     "learning_rate": config["lr"], "seed": config["seed"],
                     "test_accuracy": result.get("accuracy"),
                     "balanced_accuracy": result.get("balanced_accuracy"),
                     "test_loss": result.get("test_loss"),
                     "duration_seconds": result.get("elapsed_seconds")})
    return pd.DataFrame(rows)


def show_details(run):
    config = json.loads(run["config_json"])
    st.write(f"**{config['name'] or run['id']}** · {config['dataset'].upper()} · {run['status']}")
    if run["result_json"]:
        result = json.loads(run["result_json"])
        a, b, c = st.columns(3)
        a.metric("Test accuracy", f"{result['accuracy']:.2%}")
        b.metric("Balanced accuracy", f"{result['balanced_accuracy']:.2%}")
        c.metric("Test loss", f"{result['test_loss']:.4f}")
        st.caption(f"Train: {result['train_count']:,} · Test: {result['test_count']:,} · "
                   f"Duration: {result['elapsed_seconds']:.1f}s · Device: {result['device']}")
        st.write("Confusion matrix — rows: true class; columns: predicted class")
        matrix = pd.DataFrame(result["confusion_matrix"], index=result["class_names"], columns=result["class_names"])
        st.dataframe(matrix.style.background_gradient(cmap="Blues", axis=None), width="stretch")
        with st.expander("Class distribution"):
            st.dataframe(pd.DataFrame({"class": result["class_names"],
                                       "train": result["train_class_counts"],
                                       "test": result["test_class_counts"]}), hide_index=True)
        st.download_button("Download full result JSON", run["result_json"],
                           file_name=f"{run['id']}_result.json", mime="application/json", key=f"json_{run['id']}")
    epochs = get_epochs(run["id"])
    if epochs:
        frame = pd.DataFrame(epochs).set_index("epoch")
        left, right = st.columns(2)
        left.line_chart(frame[["loss"]], x_label="Epoch", y_label="Training loss")
        right.line_chart(frame[["accuracy"]], x_label="Epoch", y_label="Training accuracy")
    with st.expander("Saved configuration"):
        st.json(config)
    if run["error"]:
        st.error(run["error"])
    log = RUNS_DIR / run["id"] / "worker.log"
    if log.exists():
        with st.expander("Training log"):
            st.code(log.read_text(encoding="utf-8", errors="replace")[-16000:], language="text")


with st.sidebar:
    st.header("New experiment")
    dataset = st.selectbox("Dataset", ["bci", "physionet"], format_func=lambda v: "BCI IV 2a" if v == "bci" else "PhysioNet · imagery only")
    st.caption("Classes: " + ", ".join(CLASSES[dataset]))
    if dataset == "physionet":
        st.caption("Includes rest periods from imagery runs. Loading requires several GiB of RAM.")
    with st.form("experiment_form"):
        name = st.text_input("Experiment name", placeholder="Optional descriptive name", max_chars=120)
        epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=5, step=1)
        batch_size = st.selectbox("Batch size", [1, 2, 4, 8, 16, 32, 64], index=3)
        lr = st.number_input("Learning rate", min_value=0.0000001, max_value=1.0,
                             value=0.0005, step=0.0001, format="%.7f")
        seed = st.number_input("Random seed", min_value=0, max_value=4294967295, value=42, step=1)
        save_weights = st.checkbox("Save trained model weights", value=False)
        submitted = st.form_submit_button("Start experiment", type="primary", width="stretch")
    st.caption("Fixed 80/20 stratified trial split · training-only channel z-score · "
               "AdamW · full trials · one active experiment at a time.")
    database = ROOT / "datasets" / f"{dataset}_trials.sqlite"
    if not database.exists():
        st.error(f"Missing database: {database.name}")
    if submitted:
        try:
            config = ExperimentConfig(dataset=dataset, epochs=int(epochs), batch_size=batch_size,
                                      lr=float(lr), seed=int(seed), name=name, save_weights=save_weights)
            st.session_state["selected_run"] = start_run(config)
            st.success("Experiment started")
        except Exception as exc:
            st.error(str(exc))

st.info("This version uses a trial split, so subjects can appear in both sets. "
        "Use these runs for exploration; reserve separate test data before selecting final thesis settings.")
live, history = st.tabs(["Live experiment", "Compare saved results"])


@st.fragment(run_every="2s")
def live_panel():
    recover_interrupted_runs()
    runs = list_runs()
    active = next((run for run in runs if run["status"] in ACTIVE), None)
    selected = get_run(st.session_state.get("selected_run", ""))
    run = active or selected or (runs[0] if runs else None)
    if run is None:
        st.write("Choose a dataset and training parameters, then start your first experiment.")
        return
    progress = json.loads(run["progress_json"])
    if run["status"] in ACTIVE:
        stage = progress.get("stage", "queued").replace("_", " ").title()
        epoch = progress.get("epoch")
        text = stage + (f" · epoch {epoch}/{json.loads(run['config_json'])['epochs']}" if epoch else "")
        st.progress(min(float(progress.get("fraction", 0)), 1.0), text=text)
        if progress.get("batch"):
            st.caption(f"Batch {progress['batch']}/{progress['batches']}")
        if st.button("Cancel experiment", disabled=run["status"] == "cancelling"):
            cancel_run(run["id"])
            st.rerun(scope="fragment")
        if run["status"] == "cancelling":
            st.caption("Cancellation requested. Waiting for the current loading step or batch to finish.")
    elif run["status"] == "completed":
        st.success("Experiment completed and saved")
    elif run["status"] == "cancelled":
        st.warning("Experiment cancelled; partial training history was retained.")
    show_details(run)


with live:
    live_panel()

with history:
    st.button("Refresh saved results")
    runs = list_runs()
    dataset_filter = st.selectbox("Filter results by dataset", ["all", "bci", "physionet"])
    runs = [r for r in runs if dataset_filter == "all" or json.loads(r["config_json"])["dataset"] == dataset_filter]
    if runs:
        frame = comparison_frame(runs)
        st.dataframe(frame, hide_index=True, width="stretch", column_config={
            "test_accuracy": st.column_config.NumberColumn(format="%.4f"),
            "balanced_accuracy": st.column_config.NumberColumn(format="%.4f"),
        })
        st.download_button("Export results CSV", frame.to_csv(index=False), "eeg_experiments.csv", "text/csv")
        completed = frame[frame["status"] == "completed"]
        if not completed.empty:
            st.caption("Compare runs within the same dataset; their classes and difficulty differ.")
            st.bar_chart(completed.set_index("run_id")[["test_accuracy", "balanced_accuracy"]])
        choices = {r["id"]: r for r in runs}
        chosen = st.multiselect("Inspect runs side by side", list(choices), max_selections=3,
                                format_func=lambda rid: f"{json.loads(choices[rid]['config_json'])['name'] or rid} · {json.loads(choices[rid]['config_json'])['dataset']}")
        if chosen:
            for column, run_id in zip(st.columns(len(chosen)), chosen):
                with column:
                    # Unique scope because the live panel may show the same run.
                    with st.container(key=f"compare_{run_id}"):
                        run = choices[run_id]
                        st.write(f"**{run_id}**")
                        st.json(json.loads(run["config_json"]), expanded=False)
                        if run["result_json"]:
                            result = json.loads(run["result_json"])
                            st.metric("Accuracy", f"{result['accuracy']:.2%}")
                            st.metric("Balanced accuracy", f"{result['balanced_accuracy']:.2%}")
                            st.dataframe(pd.DataFrame(result["confusion_matrix"], index=result["class_names"],
                                                      columns=result["class_names"]))
                        epoch_data = get_epochs(run_id)
                        if epoch_data:
                            st.line_chart(pd.DataFrame(epoch_data).set_index("epoch")[["loss"]])
    else:
        st.caption("No saved experiments for this selection yet.")
