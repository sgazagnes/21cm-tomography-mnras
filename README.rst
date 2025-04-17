Inferring the properties of the sources of reionization using the
morphological spectra of the ionized regions
=================================================================


This repository is providing the data and notebookss needed to reproduce the results of the paper I published in the Monthly Notices of the Astronomical Society in 2021. 
The paper is available in the repo, or online

- `Inferring the properties of the sources of reionization using the
morphological spectra of the ionized regions <https://academic.oup.com/mnras/article/502/2/1816/6102530?login=false>`_ 



Paper Summary
=============

This paper presents a novel method to infer the properties of the sources that drove the **Epoch of Reionization (EoR)**  
using the morphological characteristics of ionized regions in the 21-cm signal, rather than relying solely on the traditional power spectrum.

Key Points
----------

**Background:**  
The EoR marks a phase where early galaxies ionized the neutral hydrogen in the universe.  
The 21-cm signal from hydrogen can probe this era. Upcoming instruments like **SKA** will soon enable direct imaging of these fluctuations.

**Problem:**  
Traditional 21-cm analyses use the **power spectrum**, which misses important **non-Gaussian features** like shape and topology.

**Solution Proposed:**  
The authors develop a **Bayesian inference** approach using **morphological pattern spectra**  
(size, elongation, flatness, sparseness, and compactness) of ionized regions.

**Methodology:**

- Extend the MCMC tool **21CMMC** to include morphological statistics.
- Use **21CMFAST** for simulating reionization scenarios.
- Extract morphological descriptors using a custom tool called **DISCCOFAN**.
- Analyze mock **SKA1-Low** observations with realistic noise and resolution.

Result figures:
---------------

The Python notebooks were developed to analyse the MCMC chains and plot the paper mains figures shown here. This is only meant as an overview, see the paper for more context. 

1. Bubble statistics as a function of time and instrumentation impact:

   .. raw:: html

      <div style="display: flex; justify-content: space-between;">
          <img src="Images/30_sim.png" width="45%" />
          <img src="Images/30_obs.png" width="45%" />
      </div>

2. Segmentation of ionized regions on noisy data:

   .. raw:: html

      <div style="display: flex; justify-content: space-between;">
          <img src="Images/image_cube.png" width="90%" />
      </div>

3. MCMC inference comparison:

   .. raw:: html

      <div style="display: flex; justify-content: space-between;">
          <img src="Images/30_hii_ps.png" width="32%" />
          <img src="Images/200_hii_ps.png" width="32%" />
          <img src="Images/30_hii_ps_gausse.png" width="32%" />
      </div>


4. Inference results compared to physical properties:

   .. raw:: html

      <div style="display: flex; justify-content: space-between;">
          <img src="Images/Nion1.png" width="60%" />
      </div>



Notes:
------
`The 21cmFAST code is available here <https://github.com/andreimesinger/21cmFAST>`_ 
`The 21CMMC code is available here <https://github.com/21cmfast/21CMMC>`_

Please note this project has been suuccessfully completed in 2020, the Python and inference code have not been updated since then.

AUTHOR
------

- Simon Gazagnes <sgsgazagnes@gmail.com>
