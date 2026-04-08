# napari-cellspots

A napari plugin to analyse microscopy images of cells containing spot-like features.

> [!WARNING]
> **This plugin is not intended for general use.** It is developed for a specific research workflow and may not work outside of that context. Use at your own risk.


## Features

- **ICS/IDS file reader** — opens files via the pyics library directly in napari
- **Processing widget** — browse and select ICS/IDS files, run 2D cell and spot segmentation, and visualise results coloured by nucleus ID or distance to nucleus. Corect cell segmentation manually and compute basic distribution statistics of detected spots per nucleus.
- **Polar plot widget** — visualise detected spot distributions in a polar coordinate plot per nucleus

## Installation

### pixi

You can use pixi to simplify the instatllation of the plugin and all extensions. For this:
1. Install pixi: https://pixi.prefix.dev/latest/installation/
2. Create folder and copy the [`pixi.toml`](pixi.toml) and [`pixi.lock`](pixi.lock) files from this repository into it (you can clone the repo or just download the two files).
3. Open a terminal, move to that folder and type `pixi install --all --frozen` to install the plugin and all dependencies.
4. In the current location, launch napari with `pixi run -e cellspots --frozen napari` or Jupyter with `pixi run -e cellspots --frozen jupyter lab`.
5. Alternatively, you can launch from any other location. For example to launch Jupyter Lab, type `pixi run --manifest-path /path/to/pixi.toml -e cellspots --frozen jupyter lab`.

### conda
First create an environment for napari. For example with conda:

```bash
conda create -n napari-cellspots python=3.12 napari pyqt -c conda-forge
conda activate napari-cellspots
```

Then install via pip using:

```bash
pip install git+https://github.com/guiwitz/napari-cellspots.git
```

The ICS reader is licensed under GPL-3.0, so you will need to install the pyics library separately:

```bash
pip install pyics --find-links https://github.com/guiwitz/pyics/releases/expanded_assets/v1.0.0
```

## Author

This plugin was developed by [Guillaume Witz](https://github.com/guiwitz). Parts of the code (napari widgets and simulation) were developed using Claude Sonnet 4.6. The whole code was reviewed and edited by Guillaume Witz.