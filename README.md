# HERALDO
## Holography with Extended References And Linear Differential Operator
#### Implementation written by Angus Laurenson, previous implementations, and original idea developed by the following:
* Feodor Ogrin
* Nick Bukin
* Erick Burgos Parra
* Guillaume Beutier
* Maxime Dupraz

### Abstract
This is a small module for the reconstruction of magnetic contrast from x-ray diffraction patterns. Left and right handed polarised x-rays are incident on a magnetic sample that is located in an aperture in a mask. The mask also has a slit, which acts as an extended reference. X-rays go through the aperture and the reference, before being detected by CCD which measures the intensity. Left and right handed x-rays produce different diffraction patterns due to XMCD (X-ray circular magnetic dichroism). Therefore the difference in diffraction patterns is due to the projection  of magnetic moment that is colinear with the incident x-ray beam. The diffraction pattern measured is composed of the autocorrelation of the aperture and slit, as well as the cross correlation of the slit with the aperture and the aperture with the slit. To uncorrelate the aperture and the slit a 'linear differential operator', or gradient, is applied along the slit orientation and the magnetic contrast of the sample is recovered. 
