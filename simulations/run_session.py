#!/usr/local/bin/python3.14
"""
Guided session for the language-model comparison: removes manual file naming and saving.

For one model, the script goes through the runs of run_log.csv whose answer file does not
exist yet. For each run it:
  1. copies the prompt to the Windows clipboard (WSL: through PowerShell);
  2. waits while you paste it into a NEW private chat and wait for the complete answer;
  3. reads the answer from the clipboard after you click the chat's copy button and press Enter;
  4. checks it: not the prompt itself, not identical to an answer already saved, number of
     patient codes found against the number expected (from the prompt);
  5. saves it under the right name in responses/ (never overwriting) and fills in the run log
     (date, model version, mode, web search, continued, notes).
If the answer is incomplete (the model stopped), type c to append the continuation: ask the
model to continue in the same chat, copy that part, press Enter; the parts are joined in one file.

Usage (from the llm_comparison folder):
  python3.14 scripts/run_session.py --model claude --version "Claude Opus 4.x" --mode "extended thinking on"
  python3.14 scripts/run_session.py --model grok --only synthetic        # synthetic prompts only
  python3.14 scripts/run_session.py --model deepseek --only myo5b --no-clipboard
  python3.14 scripts/run_session.py --model gemini --attach      # prompt uploaded as a file
Options:
  --only myo5b | synthetic   restrict to one part
  --no-clipboard             no PowerShell: the prompt path is shown, and the answer is pasted
                             into the terminal, ended by a line containing only END
  --attach                   for interfaces that truncate long pastes (Gemini cuts at about
                             32,000 characters): the prompt is uploaded as a file. The script
                             first copies the file's Windows path (paste it into the upload
                             dialog's file-name box), then a one-line message to send with it
Stop at any time with q (or Ctrl-C); the next session resumes at the first missing answer.
"""
import argparse
import csv
import datetime as dt
import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CODE = re.compile(r'^\s*"?(P\d{3})"?\s*[,;\t]', re.M)
N_EXPECTED = re.compile(r"Group the following (\d+) patients")
ATTACH_MSG = "Follow the instructions in the attached file."
LOG_FIELDS = ["model", "condition", "run", "prompt_file", "response_file", "date", "model_version_shown",
              "mode", "web_search", "private_chat", "continued", "notes"]


# ---------------------------------------------------------------------------------------
# Windows clipboard from WSL
# ---------------------------------------------------------------------------------------
def ps(cmd):
    return subprocess.run(["powershell.exe", "-NoProfile", "-Command", cmd], capture_output=True)


def win_path(path):
    return subprocess.run(["wslpath", "-w", str(Path(path).resolve())], capture_output=True, text=True).stdout.strip()


def to_clipboard(path):
    r = ps(f"Get-Content -Raw -Encoding UTF8 -LiteralPath '{win_path(path)}' | Set-Clipboard")
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode(errors="replace"))


def text_to_clipboard(text):
    r = subprocess.run(["powershell.exe", "-NoProfile", "-Command", "$input | Set-Clipboard"],
                       input=text.encode("utf-8"), capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode(errors="replace"))


def from_clipboard():
    r = ps("[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; Get-Clipboard -Raw")
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode(errors="replace"))
    return r.stdout.decode("utf-8", errors="replace").replace("\r\n", "\n")


def from_terminal():
    print("    Paste the answer, then a line with only END:")
    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line + "\n")
    return "".join(lines)


# ---------------------------------------------------------------------------------------
def ask(msg, choices):
    while True:
        a = input(msg).strip().lower()
        if a in choices:
            return a
        print(f"    please answer {'/'.join(choices)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="short name as in run_log.csv: claude, chatgpt, grok, gemini, deepseek")
    ap.add_argument("--version", default="", help="model name/version shown by the interface")
    ap.add_argument("--mode", default="", help='e.g. "thinking on"')
    ap.add_argument("--only", choices=["myo5b", "synthetic"])
    ap.add_argument("--log", default="run_log.csv")
    ap.add_argument("--no-clipboard", action="store_true")
    ap.add_argument("--attach", action="store_true", help="upload the prompt as a file instead of pasting it")
    a = ap.parse_args()

    use_clip = not a.no_clipboard and shutil.which("powershell.exe") is not None
    if not a.no_clipboard and not use_clip:
        print("powershell.exe not found: using --no-clipboard mode.")
    log = list(csv.DictReader(open(a.log, encoding="utf-8")))
    synthetic = lambda r: r["prompt_file"].startswith("prompts/synthetic/")
    todo = [r for r in log if r["model"] == a.model and not Path(r["response_file"]).exists()
            and (a.only is None or (a.only == "synthetic") == synthetic(r))]
    if not todo:
        sys.exit(f"Nothing left to run for {a.model}" + (f" ({a.only})" if a.only else "") + ".")
    Path("responses").mkdir(exist_ok=True)
    saved = {hashlib.md5(p.read_bytes()).hexdigest(): p.name for p in Path("responses").glob("*.txt")}
    print(f"{a.model}: {len(todo)} runs to do. For each: open a NEW private chat, paste (Ctrl-V), send, "
          f"wait for the complete answer, click its copy button, come back here.\n")

    def write_log():
        tmp = Path(tempfile.mktemp(dir=".", suffix=".csv"))
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=LOG_FIELDS, extrasaction="ignore", lineterminator="\n")
            w.writeheader(); w.writerows(log)
        tmp.replace(a.log)

    for i, r in enumerate(todo, 1):
        prompt = Path(r["prompt_file"])
        text_prompt = prompt.read_text(encoding="utf-8")
        n_exp = int(N_EXPECTED.search(text_prompt).group(1))
        out = Path(r["response_file"])
        print(f"[{i}/{len(todo)}] {prompt.name}  ->  {out.name}  ({n_exp} patients)")
        if a.attach:
            wp = win_path(prompt)
            if use_clip:
                text_to_clipboard(wp)
                print("    In a new private chat, open the upload (+) dialog and paste into its file-name box:")
            else:
                print("    In a new private chat, upload this file:")
            print(f"      {wp}")
            if ask("    When the file is attached, press Enter (q = quit): ", ["", "q"]) == "q":
                print("Stopped; the next session resumes at the first missing answer.")
                return
            if use_clip:
                text_to_clipboard(ATTACH_MSG)
                print(f'    Message copied: paste it next to the attached file and send ("{ATTACH_MSG}").')
            else:
                print(f'    Type this message next to the attached file and send: "{ATTACH_MSG}"')
        elif use_clip:
            to_clipboard(prompt)
            print("    Prompt copied to the clipboard. Paste it into a new private chat.")
        else:
            print(f"    Open {prompt} and paste its whole content into a new private chat.")
        go = ask("    When the answer is complete and copied, press Enter (s = skip this run, q = quit): ",
                 ["", "s", "q"])
        if go == "q":
            print("Stopped; the next session resumes at the first missing answer.")
            return
        if go == "s":
            print("    Skipped.\n")
            continue
        # read and check the answer (with optional continuations)
        parts, continued, action = [], False, None
        while True:
            ans = from_clipboard() if use_clip else from_terminal()
            if ans.strip() in (text_prompt.strip(), ATTACH_MSG, win_path(prompt) if a.attach else None):
                print("    The clipboard still holds the prompt: copy the model's answer first.")
                input("    Press Enter when copied. ")
                continue
            parts.append(ans)
            body = "\n".join(parts)
            codes = CODE.findall(body)
            n_found, n_unique = len(codes), len(set(codes))
            h = hashlib.md5(body.encode("utf-8")).hexdigest()
            print(f"    {n_unique} distinct patient codes found (expected {n_exp})"
                  + (f", {n_found - n_unique} repeated" if n_found > n_unique else ""))
            if h in saved:
                print(f"    WARNING: identical to {saved[h]} already saved: wrong chat copied?")
            if n_unique < n_exp:
                action = ask("    Incomplete. c = add a continuation, s = save as it is, r = recopy the answer, "
                             "x = skip this run: ", ["c", "s", "r", "x"])
            else:
                action = ask("    Save? Enter = yes, r = recopy, x = skip this run: ", ["", "r", "x"]) or "s"
            if action == "c":
                continued = True
                input("    Ask the model to continue in the same chat, copy the continuation, press Enter. ")
                continue
            if action == "r":
                parts.pop()
                input("    Copy the answer again, press Enter. ")
                continue
            break
        if action == "x":
            print("    Skipped.\n")
            continue
        if out.exists():                                     # never overwrite
            sys.exit(f"{out} appeared meanwhile; not overwritten. Stopping.")
        out.write_text(body, encoding="utf-8")
        saved[hashlib.md5(body.encode('utf-8')).hexdigest()] = out.name
        web = ask("    Did the interface show any web search or opened page? y/n: ", ["y", "n"])
        note = input("    Note (optional, Enter for none): ").strip()
        mode = "; ".join(x for x in [a.mode or r.get("mode", ""), "prompt attached as file" if a.attach else ""] if x)
        r.update(date=dt.date.today().isoformat(), model_version_shown=a.version or r.get("model_version_shown", ""),
                 mode=mode, web_search="used" if web == "y" else "none",
                 continued="yes" if continued else "no",
                 notes="; ".join(x for x in [r.get("notes", ""), note,
                                             f"{n_unique}/{n_exp} codes" if n_unique < n_exp else ""] if x))
        write_log()
        print(f"    Saved {out} and updated {a.log}.\n")
        if web == "y":
            print("    Web search used: rerun this prompt later (rename or move the file first).\n")
    print("Session finished.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped; answers saved so far are kept, the next session resumes at the first missing one.")
