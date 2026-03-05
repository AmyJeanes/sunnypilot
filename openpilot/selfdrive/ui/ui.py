#!/usr/bin/env python3
import os
import time

import pyray as rl
from openpilot.cereal import messaging
from openpilot.common.hardware import COMMA_HARDWARE
from openpilot.common.realtime import Priority, config_realtime_process, set_core_affinity
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.layouts.main import MainLayout
from openpilot.selfdrive.ui.mici.layouts.main import MiciMainLayout
from openpilot.selfdrive.ui.ui_state import ui_state

BIG_UI = gui_app.big_ui()


def main():
  cores = {5, }
  # above plannerd and radard
  config_realtime_process(0, Priority.CTRL_HIGH)

  gui_app.init_window("UI")
  def _build_layout(big: bool):
    layout = MainLayout() if big else MiciMainLayout()
    layout.set_rect(rl.Rectangle(0, 0, gui_app.width, gui_app.height))
    return layout

  current_big_ui = gui_app.big_ui()
  main_layout = _build_layout(current_big_ui)
  pm = messaging.PubMaster(['uiDebug'])
  for should_render, frame_time, cpu_time in gui_app.render():
    extra_start = time.monotonic()
    ui_state.update()

    # Rebuild layout when the compact UI toggle changes while disengaged
    desired_big_ui = gui_app.big_ui()
    if not ui_state.engaged and desired_big_ui != current_big_ui:
      gui_app.reset_navigation()
      gui_app.resize_for_layout(desired_big_ui)
      main_layout = _build_layout(desired_big_ui)
      current_big_ui = desired_big_ui

    if should_render:
      # reaffine after power save offlines our core
      if COMMA_HARDWARE and os.sched_getaffinity(0) != cores:
        try:
          set_core_affinity(list(cores))
        except OSError:
          pass

      extra_cpu = time.monotonic() - extra_start
      msg = messaging.new_message('uiDebug')
      msg.uiDebug.cpuTimeMillis = (cpu_time + extra_cpu) * 1000
      msg.uiDebug.frameTimeMillis = frame_time * 1000
      pm.send('uiDebug', msg)


if __name__ == "__main__":
  main()
