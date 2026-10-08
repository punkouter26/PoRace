using System;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

namespace PoRace {

/// <summary>Pre-race screen (PRD 4.1): pick a map discovered by MapRegistry, the number of racers (capped by the
/// map's grid) and the laps, then launch. Headless: PoRace.exe -batchmode -nographics -menuMap 1 -raceTest ...</summary>
public class MenuController : MonoBehaviour {
  public Text mapText, infoText, racersText, lapsText;
  MapDefinition[] _maps; int _i, _racers = 4, _laps = 1;

  void Start() {
    _maps = MapRegistry.All();
    if (_maps.Length == 0) { if (mapText) mapText.text = "No maps in Resources/Maps"; return; }
    _racers = RaceConfig.fromMenu ? RaceConfig.racers : 4;
    SelectMap(0);
    var args = Environment.GetCommandLineArgs();
    for (int i = 0; i + 1 < args.Length; i++) if (args[i] == "-menuMap") { SelectMap(int.Parse(args[i + 1])); Invoke(nameof(OnLaunch), 0.5f); }
  }

  void SelectMap(int i) {
    _i = ((i % _maps.Length) + _maps.Length) % _maps.Length;
    _laps = _maps[_i].closed ? _maps[_i].defaultLaps : 1;
    _racers = Mathf.Clamp(_racers, 1, _maps[_i].gridSlots);
    Refresh();
  }

  void Refresh() {
    var m = _maps[_i];
    if (mapText) mapText.text = m.displayName;
    if (infoText) infoText.text = $"{m.Length * (m.closed ? _laps : 1):F0} m   {m.terrain}   difficulty {new string('*', m.difficulty)}" + System.Environment.NewLine
        + $"{(m.closed ? "circuit" : "point to point")}   grid {m.gridSlots}   map {_i + 1} of {_maps.Length}";
    if (racersText) racersText.text = _racers.ToString();
    if (lapsText) lapsText.text = m.closed ? _laps.ToString() : "-";
  }

  public void OnNextMap() => SelectMap(_i + 1);
  public void OnPrevMap() => SelectMap(_i - 1);
  public void OnRacersUp() { _racers = Mathf.Min(_racers + 1, _maps[_i].gridSlots); Refresh(); }
  public void OnRacersDown() { _racers = Mathf.Max(_racers - 1, 1); Refresh(); }
  public void OnLapsUp() { if (_maps[_i].closed) _laps = Mathf.Min(_laps + 1, 5); Refresh(); }
  public void OnLapsDown() { if (_maps[_i].closed) _laps = Mathf.Max(_laps - 1, 1); Refresh(); }

  public void OnLaunch() {
    RaceConfig.racers = _racers; RaceConfig.laps = _laps; RaceConfig.seed = -1; RaceConfig.fromMenu = true;
    SceneManager.LoadScene(_maps[_i].sceneName);
  }
}

}
