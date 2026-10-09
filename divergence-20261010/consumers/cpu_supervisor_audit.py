"""Small CPU contracts for supervisory ownership facts; synthetic data only."""
import importlib.util
import json
from pathlib import Path
import tempfile

root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("supervisor_audit", root / "supervisor_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
manifest = root / "frozen-producer-v6/producer-manifest.json"
when = "2026-10-10T02:00:00Z"
cid = "a" * 64
owner = "synthetic-contract-only"
first, second = audit.GPUS.values()
with tempfile.TemporaryDirectory(prefix="supervisor-cpu-", dir=root) as name:
    work = Path(name)
    assert work.resolve().parent == root
    def put(file, text):
        (work / file).write_text(text, encoding="utf-8")
    put("container-id.txt", cid)
    for file in ("matrix.exit", "short-A.exit", "short-B.exit", "short-C.exit", "cleanup-stop.exit"):
        put(file, "0\n")
    put("monitor-final.exit", "143\n")
    for phase in ("initial", "before", "after"):
        state = {"Running": phase != "after", "Status": "exited" if phase == "after" else "running",
                 "Pid": 0 if phase == "after" else 99, "OOMKilled": False,
                 "Dead": False, "Error": "", "FinishedAt": when if phase == "after" else ""}
        put("container-inspect-" + phase + ".json", json.dumps([
            {"Id": cid, "Config": {"Labels": {"codex.owner": owner}}, "State": state}]))
    put("host-identities.log", when + "|" + cid + "|" + owner + "|true\n")
    put("host-gpus.csv", when + ",4," + first + ",20000 MiB,50 %,40,1500 MHz,P0\n" +
        when + ",6," + second + ",20000 MiB,50 %,40,1500 MHz,P0\n")
    complete = when + "\n" + first + ",101,100 MiB\nOWNED\n101\n99\n"
    put("host-processes.log", complete)
    put("final-gpus.csv", "index, uuid, memory.used [MiB], utilization.gpu [%]\n" +
        "4," + first + ",1 MiB,0 %\n6," + second + ",1 MiB,0 %\n")
    put("final-processes.csv", "gpu_uuid, pid, used_gpu_memory [MiB]\n")
    passed = audit.audit(work, manifest, owner)
    assert passed["status"] == "PASS" and passed["strict_ownership"]
    put("host-processes.log", complete.replace("101,100", "102,100"))
    unmatched = audit.audit(work, manifest, owner)
    assert unmatched["status"] == "REVIEW_REQUIRED" and not unmatched["strict_ownership"]
    assert unmatched["counts"]["unmatched"] == 1
    put("host-processes.log", complete)
    put("repeated-unmatched-owner.txt", when)
    failed = audit.audit(work, manifest, owner)
    assert failed["status"] == "FAIL" and not failed["strict_ownership"]
    (work / "repeated-unmatched-owner.txt").unlink()
    put("final-processes.csv", "gpu_uuid, pid, used_gpu_memory [MiB]\n" + first + ",101,100 MiB\n")
    final_unmatched = audit.audit(work, manifest, owner)
    assert final_unmatched["status"] == "FAIL" and not final_unmatched["strict_ownership"]
    assert final_unmatched["counts"]["unmatched"] == 1
    assert final_unmatched["details"]["final_compute_observations"][0]["classification"] == "unmatched_after_container_stop"
    partial = audit.process_frames(when + "\n" + first + ",101,100 MiB\n")
    assert not partial[0]["owned_section_seen"] and len(partial[0]["compute"]) == 1
    try:
        audit.inspection({"Running": False}, cid, owner)
        raise AssertionError("State-only must not prove identity")
    except ValueError:
        pass
print(json.dumps({"scope": "CPU synthetic parser/ownership contracts only", "gpu_execution": False,
                  "contracts": ["matched", "unmatched-retained", "marker-fails", "final-PID-retained", "partial-retained", "state-only-rejected"],
                  "status": "PASS"}))
