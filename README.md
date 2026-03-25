# napari-cellspots

A napari plugin to analyse microscopy images of cells containing spot-like features.

> [!WARNING]
> **This plugin is not intended for general use.** It is developed for a specific research workflow and may not work outside of that context. Use at your own risk.


## Features

- **ICS/IDS file reader** — opens files via the pyics library directly in napari
- **Processing widget** — browse and select ICS/IDS files, run 2D cell and spot segmentation, and visualise results coloured by nucleus ID or distance to nucleus
- **Polar plot widget** — visualise detected spot distributions in a polar coordinate plot per nucleus

## Installation

First create an environment for napari. For example with conda:

```bash
conda create -n napari-cellspots python=3.13 napari pyqt -c conda-forge
conda activate napari-cellspots
```

Then install via pip using:

```bash
pip install git+https://github.com/guiwitz/napari-cellspots.git
```

## Author

This plugin was developed by [Guillaume Witz](https://github.com/guiwitz). Parts of the code (napari widgets and simulation) were developed using Claude Sonnet 4.6. The whole code was reviewed and edited by Guillaume Witz.