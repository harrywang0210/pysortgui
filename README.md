# PySortGUI: A python-based open-source GUI software for spike sorting.
PySortGUI provides a graphical user interface for importing, preprocessing, and visualizing neural signal data. It also integrates autosorting algorithm and manual clustering.

![Image](https://github.com/HarryWang0210/PySortGUI/blob/master/screenshots/2.1Overview.png)

## Installation instructions
Run the following commands in a terminal:
1. Create a new conda environment: `conda create -n pysortgui -y python=3.12`
2. Activate the new conda environment with `conda activate pysortgui`
3. Install pysortgui: `pip install git+https://github.com/HarryWang0210/PySortGUI.git`

## Running PySortGUI
Running with gui: `pysortgui`  
Running without gui for bulk processing: `pysortgui-cli {command}`

## Abstract
Spike sorting plays an important role in understanding brain activity mechanisms. While extracellular recordings can capture the multiple neuron activities simultaneously, they face the challenge of signal mixing, which needs to be resolved through spike sorting. Despite numerous studies aimed at improving the accuracy of automatic spike sorting, the results of fully automatic algorithms are still not entirely convincing and require further manual verification and adjustment. To perform spike sorting with automatic algorithms and manual clustering, Dr. Alessandro Scaglione, a postdoctoral research fellow under Professor Shih-Chieh Lin at our Institute of Neuroscience, previously developed the software called SpikeSorterGL. However, this software is incompatible with modern systems and hard to maintain. To address this, we have developed an open-source graphical user interface software, PySortGUI, to replace and improve the SpikeSorterGL. Our objectives include enhancing system compatibility, optimizing core functionalities, integrating new features, and improving the user interface, thereby providing a more effective spike sorting tool for neuroscience research.  
[Full text](https://hdl.handle.net/11296/76dcat)
