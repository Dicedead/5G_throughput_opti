COM-500 Mini-Project: Group Throughput optimization in Wireless Communications via Spatial Beamforming
======================================================================================================

`data_1.npz` and `data_2.npz` contain data products from two different 5G base-stations.

* XYZ: (N_antenna, 2) cartesian antenna positions
* S: (N_slot, N_antenna, N_antenna) covariance matrices per time slot.
* T_slot: duration of a slot [s]
* lambda_: channel wavelength [m]
* src_model: (N_tx, 2) unit directions of transmitters in the cell area.

Communication systems process data at high throughput. As such it is difficult to capture long antenna traces.
Since throughput assessment via beamforming only requires solving positional inferance problems, we instead provide the (much) smaller covariance estimates between antennas per short time interval (1 [ms]).
