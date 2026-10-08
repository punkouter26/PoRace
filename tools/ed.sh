#!/usr/bin/env bash
# Helpers for driving the Unity editor from Git Bash. Source this file.
ucmd() { unity command "$@" --caller plugin --skill unity-cli --format json --no-banner 2>&1; }
recompile() {
  unity command recompile --caller plugin --skill unity-cli --no-banner >/dev/null 2>&1
  local st=""
  for i in $(seq 1 40); do sleep 5
    st=$(ucmd recompile_status | python -c "import sys,json; r=json.load(sys.stdin)['data']['result']; print(r.get('status'), r.get('errors'))" 2>/dev/null)
    case "$st" in completed*) break;; esac
  done
  echo "compile: $st"; sleep 6
  [ "$st" = "completed []" ]
}
ueval() {  # ueval 'C# statement;'
  unity command eval --caller plugin --skill unity-cli --format json --no-banner --timeout 240 -- "$1" 230000 2>&1 | python -c "
import sys,json
d=json.load(sys.stdin); r=(d.get('data') or {}).get('result') or {}
print('eval:', r.get('success'), r.get('diagnostics') or '', d.get('errors') or '')"
}
ulog() {  # ulog 'substring'
  ucmd console | python -c "
import sys,json
k=sys.argv[1]; d=json.load(sys.stdin); es=[x for x in d['data']['result']['entries'] if k in str(x['message']) or x['level']=='error']
[print(x['level'], str(x['message'])[:280].replace(chr(10),' | ')) for x in es[-4:]]" "$1"
}
ubuild() {  # ubuild Build/X/PoRace.exe '["Assets/Scenes/A.unity"]'
  unity command build --caller plugin --skill unity-cli --format json --no-banner -- StandaloneWindows64 "$1" "" "" "$2" true false >/dev/null 2>&1
  local st=""
  for i in $(seq 1 60); do sleep 10
    st=$(ucmd build_status | python -c "import sys,json; print(json.load(sys.stdin)['data']['result'].get('status'))" 2>/dev/null)
    [ "$st" = "completed" ] && break
  done
  echo "$1 -> $st"
}
