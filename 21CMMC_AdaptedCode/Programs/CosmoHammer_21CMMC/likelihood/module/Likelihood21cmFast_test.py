#!/usr/bin/env python

import os
import numpy as np
np.seterr(invalid='ignore', divide='ignore')
from decimal import *
from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline
import string
import subprocess
import time
import multiprocessing
import skimage
from astropy.io import fits  # We use fits to open the actual data file
import tools21cm as t2c
import scipy 
import math
from scipy.misc import factorial
from scipy.stats import poisson
import time
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import Voronoi, Delaunay, delaunay_plot_2d,voronoi_plot_2d
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree
from random import randrange


TWOPLACES = Decimal(10) ** -2       # same as Decimal('0.01')
FOURPLACES = Decimal(10) ** -4       # same as Decimal('0.0001')
SIXPLACES = Decimal(10) ** -6       # same as Decimal('0.000001')

McGreer_Redshift = 5.9

# The redshift of the QSO
QSO_Redshift = 7.0842

def load_binary_data(filename, dtype=np.float32): 
     """ 
     We assume that the data was written 
     with write_binary_data() (little endian). 
     """ 
     f = open(filename, "rb") 
     data = f.read() 
     f.close() 
     _data = np.fromstring(data, dtype) 

     return _data

def LogFactorial(n):
    if (n > 254):
        x = n + 1;
        return (x - 0.5)*np.log(x) - x + 0.5*np.log(2*3.14) + 1.0/(12.0*x)    
    else:
        lf = [0.000000000000000,\
            0.000000000000000,\
            0.693147180559945,\
            1.791759469228055,\
            3.178053830347946,\
            4.787491742782046,\
            6.579251212010101,\
            8.525161361065415,\
            10.604602902745251,\
            12.801827480081469,\
            15.104412573075516,\
            17.502307845873887,\
            19.987214495661885,\
            22.552163853123421,\
            25.191221182738683,\
            27.899271383840894,\
            30.671860106080675,\
            33.505073450136891,\
            36.395445208033053,\
            39.339884187199495,\
            42.335616460753485,\
            45.380138898476908,\
            48.471181351835227,\
            51.606675567764377,\
            54.784729398112319,\
            58.003605222980518,\
            61.261701761002001,\
            64.557538627006323,\
            67.889743137181526,\
            71.257038967168000,\
            74.658236348830158,\
            78.092223553315307,\
            81.557959456115029,\
            85.054467017581516,\
            88.580827542197682,\
            92.136175603687079,\
            95.719694542143202,\
            99.330612454787428,\
            102.968198614513810,\
            106.631760260643450,\
            110.320639714757390,\
            114.034211781461690,\
            117.771881399745060,\
            121.533081515438640,\
            125.317271149356880,\
            129.123933639127240,\
            132.952575035616290,\
            136.802722637326350,\
            140.673923648234250,\
            144.565743946344900,\
            148.477766951773020,\
            152.409592584497350,\
            156.360836303078800,\
            160.331128216630930,\
            164.320112263195170,\
            168.327445448427650,\
            172.352797139162820,\
            176.395848406997370,\
            180.456291417543780,\
            184.533828861449510,\
            188.628173423671600,\
            192.739047287844900,\
            196.866181672889980,\
            201.009316399281570,\
            205.168199482641200,\
            209.342586752536820,\
            213.532241494563270,\
            217.736934113954250,\
            221.956441819130360,\
            226.190548323727570,\
            230.439043565776930,\
            234.701723442818260,\
            238.978389561834350,\
            243.268849002982730,\
            247.572914096186910,\
            251.890402209723190,\
            256.221135550009480,\
            260.564940971863220,\
            264.921649798552780,\
            269.291097651019810,\
            273.673124285693690,\
            278.067573440366120,\
            282.474292687630400,\
            286.893133295426990,\
            291.323950094270290,\
            295.766601350760600,\
            300.220948647014100,\
            304.686856765668720,\
            309.164193580146900,\
            313.652829949878990,\
            318.152639620209300,\
            322.663499126726210,\
            327.185287703775200,\
            331.717887196928470,\
            336.261181979198450,\
            340.815058870798960,\
            345.379407062266860,\
            349.954118040770250,\
            354.539085519440790,\
            359.134205369575340,\
            363.739375555563470,\
            368.354496072404690,\
            372.979468885689020,\
            377.614197873918670,\
            382.258588773060010,\
            386.912549123217560,\
            391.575988217329610,\
            396.248817051791490,\
            400.930948278915760,\
            405.622296161144900,\
            410.322776526937280,\
            415.032306728249580,\
            419.750805599544780,\
            424.478193418257090,\
            429.214391866651570,\
            433.959323995014870,\
            438.712914186121170,\
            443.475088120918940,\
            448.245772745384610,\
            453.024896238496130,\
            457.812387981278110,\
            462.608178526874890,\
            467.412199571608080,\
            472.224383926980520,\
            477.044665492585580,\
            481.872979229887900,\
            486.709261136839360,\
            491.553448223298010,\
            496.405478487217580,\
            501.265290891579240,\
            506.132825342034830,\
            511.008022665236070,\
            515.890824587822520,\
            520.781173716044240,\
            525.679013515995050,\
            530.584288294433580,\
            535.496943180169520,\
            540.416924105997740,\
            545.344177791154950,\
            550.278651724285620,\
            555.220294146894960,\
            560.169054037273100,\
            565.124881094874350,\
            570.087725725134190,\
            575.057539024710200,\
            580.034272767130800,\
            585.017879388839220,\
            590.008311975617860,\
            595.005524249382010,\
            600.009470555327430,\
            605.020105849423770,\
            610.037385686238740,\
            615.061266207084940,\
            620.091704128477430,\
            625.128656730891070,\
            630.172081847810200,\
            635.221937855059760,\
            640.278183660408100,\
            645.340778693435030,\
            650.409682895655240,\
            655.484856710889060,\
            660.566261075873510,\
            665.653857411105950,\
            670.747607611912710,\
            675.847474039736880,\
            680.953419513637530,\
            686.065407301994010,\
            691.183401114410800,\
            696.307365093814040,\
            701.437263808737160,\
            706.573062245787470,\
            711.714725802289990,\
            716.862220279103440,\
            722.015511873601330,\
            727.174567172815840,\
            732.339353146739310,\
            737.509837141777440,\
            742.685986874351220,\
            747.867770424643370,\
            753.055156230484160,\
            758.248113081374300,\
            763.446610112640200,\
            768.650616799717000,\
            773.860102952558460,\
            779.075038710167410,\
            784.295394535245690,\
            789.521141208958970,\
            794.752249825813460,\
            799.988691788643450,\
            805.230438803703120,\
            810.477462875863580,\
            815.729736303910160,\
            820.987231675937890,\
            826.249921864842800,\
            831.517780023906310,\
            836.790779582469900,\
            842.068894241700490,\
            847.352097970438420,\
            852.640365001133090,\
            857.933669825857460,\
            863.231987192405430,\
            868.535292100464630,\
            873.843559797865740,\
            879.156765776907600,\
            884.474885770751830,\
            889.797895749890240,\
            895.125771918679900,\
            900.458490711945270,\
            905.796028791646340,\
            911.138363043611210,\
            916.485470574328820,\
            921.837328707804890,\
            927.193914982476710,\
            932.555207148186240,\
            937.921183163208070,\
            943.291821191335660,\
            948.667099599019820,\
            954.046996952560450,\
            959.431492015349480,\
            964.820563745165940,\
            970.214191291518320,\
            975.612353993036210,\
            981.015031374908400,\
            986.422203146368590,\
            991.833849198223450,\
            997.249949600427840,\
            1002.670484599700300,\
            1008.095434617181700,\
            1013.524780246136200,\
            1018.958502249690200,\
            1024.396581558613400,\
            1029.838999269135500,\
            1035.285736640801600,\
            1040.736775094367400,\
            1046.192096209724900,\
            1051.651681723869200,\
            1057.115513528895000,\
            1062.583573670030100,\
            1068.055844343701400,\
            1073.532307895632800,\
            1079.012946818975000,\
            1084.497743752465600,\
            1089.986681478622400,\
            1095.479742921962700,\
            1100.976911147256000,\
            1106.478169357800900,\
            1111.983500893733000,\
            1117.492889230361000,\
            1123.006317976526100,\
            1128.523770872990800,\
            1134.045231790853000,\
            1139.570684729984800,\
            1145.100113817496100,\
            1150.633503306223700,\
            1156.170837573242400]
        return lf[n];

def voronoi_volumes(v):
    #v = Voronoi(points,qhull_options='QJ')
    vol = np.zeros(v.npoints)
    #regions, vertices = voronoi_finite_polygons_2d(v)
    #print regions, v.regions
    for i, reg_num in enumerate(v.point_region):

        indices = v.regions[reg_num]
        if -1 in indices: # some regions can be opened
            vol[i] = 1e100
        else:
            vol[i] = ConvexHull(v.vertices[indices],qhull_options='QJ').volume
            #print vol[i]
    return vol

class Likelihood21cmFast_multiz(object):
    
    def __init__(self, Redshifts_For_LF,Muv_values, phi_values, phi_Error, k_values, PS_values, Error_k_values, PS_Error, Redshift, Redshifts_For_Prior, param_legend, Fiducial_Params, FlagOptions, param_string_names, NSplinePoints,TsCalc_z, Foreground_cut, Shot_Noise_cut, IncludeLightCone, IncludeLF, ModUncert, PriorLegend, NFValsQSO, PDFValsQSO, IncludePS,IncludeMorphologyHII,IncludeMorphologyHI ,IncludePoissonBubHII,IncludePoissonBubHI, HII_DIMS, MPC_DIMS, mock_hii_dis,  mock_hii_vol, mock_hii_vor, mock_hi_dis,  mock_hi_vol, mock_hi_vor, uv_maps):
        self.Redshifts_For_LF = Redshifts_For_LF # New in v1.4
        self.Muv_values = Muv_values   # New in v1.4
        self.phi_values = phi_values   # New in v1.4
        self.phi_Error = phi_Error     # New in v1.4
        self.k_values = k_values
        self.PS_values = PS_values
        self.Error_k_values = Error_k_values
        self.PS_Error = PS_Error
        self.Redshift = Redshift
        self.Redshifts_For_Prior = Redshifts_For_Prior
        self.param_legend = param_legend
        self.Fiducial_Params = Fiducial_Params
        self.FlagOptions = FlagOptions
        self.param_string_names = param_string_names
        self.NSplinePoints = NSplinePoints
        self.TsCalc_z = TsCalc_z
        self.Foreground_cut = Foreground_cut
        self.Shot_Noise_cut = Shot_Noise_cut
        self.IncludeLightCone = IncludeLightCone
        self.IncludeLF = IncludeLF
        self.ModUncert = ModUncert
        self.PriorLegend = PriorLegend
        self.NFValsQSO = NFValsQSO
        self.PDFValsQSO = PDFValsQSO
        self.IncludePS = IncludePS
        self.IncludeMorphologyHII = IncludeMorphologyHII
        self.IncludeMorphologyHI = IncludeMorphologyHI
        self.IncludePoissonBubHII = IncludePoissonBubHII
        self.IncludePoissonBubHI = IncludePoissonBubHI
        self.HII_DIMS = HII_DIMS
        self.MPC_DIMS = MPC_DIMS
       # self.vol_hii_bins = vol_hii_bins
        self.mock_hii_dis = mock_hii_dis
        self.mock_hii_vol = mock_hii_vol
        self.mock_hii_vor = mock_hii_vor
        self.mock_hi_dis = mock_hi_dis
        self.mock_hi_vol = mock_hi_vol
        self.mock_hi_vor = mock_hi_vor
        self.uv_maps = uv_maps

    def Likelihood(self,ctx):

        params = ctx.getParams()

        # If the light-cone option is set, we do not return the neutral fraction as it can be a large amount of data (also less useful).
        # Only really helpful (if at all) for co-eval cubes
        if self.IncludeLightCone is True:
            nf_vals = np.zeros(3)
        else:

            # If we are applying the optical depth prior, then we might as well keep the value of the electron scattering optical depth
            if self.PriorLegend['PlanckPrior'] is True or self.FlagOptions['KEEP_ALL_DATA'] is True:
                nf_vals = np.zeros(len(self.Redshift) + len(self.Redshifts_For_Prior)+3)
            else:
                nf_vals = np.zeros(len(self.Redshift) + len(self.Redshifts_For_Prior)+2)

        # Generate a unique ID for each thread by sampling a randomly seeded distribution.
        # Given than file I/O needs to be unique to each thread, it is beneficial to provide a unique ID in the off chance that two different threads 
        # end up with the same walker position (same parameter set)
        np.random.seed()
        
        random_number = np.random.normal(size=1)

        # Create a second unique ID, that being the first variable of the specific walker (fail-safe against ID overlap; shouldn't happen, but guarding against anyway)
        Individual_ID = Decimal(repr(random_number[0])).quantize(SIXPLACES)
        Individual_ID_2 = Decimal(repr(params[0])).quantize(SIXPLACES)

        # Add all the redshifts (those for the likelihood and those for prior only). This parameter is only used where this is relevant
        number_redshifts = len(self.Redshift) + len(self.Redshifts_For_Prior)
        
        # Add and sort all redshifts (those for the likelihood and those for prior only)        
        AllRedshifts = []
        if self.IncludeLightCone is False:
            for i in range(len(self.Redshift)):
                AllRedshifts.append(self.Redshift[i])
    
            for i in range(len(self.Redshifts_For_Prior)):
                AllRedshifts.append(self.Redshifts_For_Prior[i])

            AllRedshifts.sort(key=float)


        StoredStatisticalData = []
        StoredStatisticalData_Error = []
        StoredFileLayout = []
        StoredFileLayout_Error = []

        separator_column = "\t"

        if self.IncludeLightCone is True:
            LightConeFlag = 1
        else:
            LightConeFlag = 0

        separator = " "
        separator_other = "_"
        seq = []
        # Add the random thread ID
        seq.append("%s"%(Individual_ID))
        # Add the second ID
        seq.append("%s"%(Individual_ID_2))

        StringArgument_other = string.join(seq,separator_other)

        # Add number of redshifts
        # If using the light-cone version of the code, don't need to set a redshift
        if self.IncludeLightCone is True:
            seq.append("0")
        else:
            seq.append("%s"%(number_redshifts))
        # Add light cone flag
        seq.append("%s"%(LightConeFlag))
        
	# If mass-dependence on ionising efficiency is allowed. Add the flag here
        if self.FlagOptions['USE_MASS_DEPENDENT_ZETA'] is True:
            seq.append("1")
        else:
            seq.append("0")
        # Add redshift for Ts.c calculation
        seq.append("%s"%(self.TsCalc_z))

        #StringArgument = string.join(seq,separator)
        #print 'StringArgument:',StringArgument
        
        #if self.IncludeLF is True:
        if self.IncludeLF is 1:
            seq.append("1")
        elif self.IncludeLF is 2:
		    seq.append("2")
        else:
            seq.append("0")

        StringArgument = string.join(seq,separator)

        ##### Now we need to create the individual walker file to be read by drive_21cmMC_streamlined #####
        
        if self.FlagOptions['GENERATE_NEW_ICS'] is True:
            GenerateNewICs = 1
        else:
            GenerateNewICs = 0

        if self.FlagOptions['INCLUDE_RSDS'] is True:
            Subcell_RSDs = 1
        else:
            Subcell_RSDs = 0

        if self.FlagOptions['USE_IONISATION_FCOLL_TABLE'] is True:
            IONISATION_FCOLL_TABLE = 1
        else:
            IONISATION_FCOLL_TABLE = 0

        if self.FlagOptions['USE_FCOLL_TABLE'] is True:
            UseFcollTable = 1
        else:
            UseFcollTable = 0

        if self.FlagOptions['CALC_TS_FLUC'] is True:
            PerformTsCalc = 1
        else:
            PerformTsCalc = 0

        if self.FlagOptions['USE_INHOMO_RECO'] is True:
            INHOMO_RECO = 1
        else:
            INHOMO_RECO = 0

        if self.FlagOptions['KEEP_GLOBAL_DATA'] is True:
            OutputGlobalAve = 1
        else:

            if self.PriorLegend['PlanckPrior'] is True or self.PriorLegend['McGreerPrior'] is True or self.PriorLegend['GreigPrior'] is True or self.FlagOptions['KEEP_ALL_DATA'] is True: 
                OutputGlobalAve = 1
            elif self.IncludeLightCone is True:
                OutputGlobalAve = 1
            else:
                OutputGlobalAve = 0

        parameter_number = 0
        create_file = open("Walker_%s.txt"%(StringArgument_other),"w")
        create_file.write("FLAGS    %s    %s    %s    %s    %s    %s    %s\n"%(GenerateNewICs,Subcell_RSDs,IONISATION_FCOLL_TABLE,UseFcollTable,PerformTsCalc,INHOMO_RECO,OutputGlobalAve))
        
        # New in v1.4
        if self.param_legend['F_STAR10'] is True:    
            create_file.write("F_STAR10    %s\n"%(Decimal(repr(params[parameter_number])).quantize(SIXPLACES)))
            parameter_number += 1
        else:
            create_file.write("F_STAR10    %s\n"%(self.Fiducial_Params['F_STAR10']))

        if self.param_legend['ALPHA_STAR'] is True:    
            create_file.write("ALPHA_STAR    %s\n"%(Decimal(repr(params[parameter_number])).quantize(SIXPLACES)))
            parameter_number += 1
        else:
            create_file.write("ALPHA_STAR    %s\n"%(self.Fiducial_Params['ALPHA_STAR']))

        if self.param_legend['F_ESC10'] is True:    
            create_file.write("F_ESC10    %s\n"%(Decimal(repr(params[parameter_number])).quantize(SIXPLACES)))
            parameter_number += 1
        else:
            create_file.write("F_ESC10    %s\n"%(self.Fiducial_Params['F_ESC10']))

        if self.param_legend['ALPHA_ESC'] is True:    
            create_file.write("ALPHA_ESC    %s\n"%(Decimal(repr(params[parameter_number])).quantize(SIXPLACES)))
            parameter_number += 1
        else:
            create_file.write("ALPHA_ESC    %s\n"%(self.Fiducial_Params['ALPHA_ESC']))

        if self.param_legend['M_TURN'] is True:    
            create_file.write("M_TURN    %s\n"%(Decimal(repr(params[parameter_number])).quantize(SIXPLACES)))
            parameter_number += 1
        else:
            create_file.write("M_TURN    %s\n"%(self.Fiducial_Params['M_TURN']))

        if self.param_legend['t_STAR'] is True:    
            create_file.write("t_STAR    %s\n"%(Decimal(repr(params[parameter_number])).quantize(SIXPLACES)))
            parameter_number += 1
        else:
            create_file.write("t_STAR    %s\n"%(self.Fiducial_Params['t_STAR']))

        if self.param_legend['ZETA'] is True:
            #create_file.write("ZETA    %s\n"%(Decimal(repr(params[parameter_number])).quantize(SIXPLACES)))
            create_file.write("ZETA    %1.6f\n"%(30.000))
            parameter_number += 1
        else:
            create_file.write("ZETA    %s\n"%(self.Fiducial_Params['ZETA']))

        if self.param_legend['MFP'] is True:
            #create_file.write("MFP    %s\n"%(Decimal(repr(params[parameter_number])).quantize(SIXPLACES)))
            create_file.write("MFP    %1.6f\n"%(15.00))

            parameter_number += 1
        else:
            create_file.write("MFP    %s\n"%(self.Fiducial_Params['MFP']))

        if self.param_legend['TVIR_MIN'] is True:
            #create_file.write("TVIR_MIN    %s\n"%(Decimal(repr(params[parameter_number])).quantize(SIXPLACES)))
            create_file.write("TVIR_MIN    %1.6f\n"%(4.70))
            X_RAY_TVIR_MIN = params[parameter_number]
            parameter_number += 1
        else:
            create_file.write("TVIR_MIN    %s\n"%(self.Fiducial_Params['TVIR_MIN']))

        if self.param_legend['L_X'] is True:
            create_file.write("L_X    %s\n"%(Decimal(repr(params[parameter_number])).quantize(SIXPLACES)))
            parameter_number += 1
        else:
            create_file.write("L_X    %s\n"%(self.Fiducial_Params['L_X']))

        if self.param_legend['NU_X_THRESH'] is True:
            create_file.write("NU_X_THRESH    %s\n"%(Decimal(repr(params[parameter_number])).quantize(SIXPLACES)))
            parameter_number += 1
        else:
            create_file.write("NU_X_THRESH    %s\n"%(self.Fiducial_Params['NU_X_THRESH']))

        create_file.write("NU_X_BAND_MAX    %s\n"%(self.Fiducial_Params['NU_X_BAND_MAX']))
        create_file.write("NU_X_MAX    %s\n"%(self.Fiducial_Params['NU_X_MAX']))

        if self.param_legend['X_RAY_SPEC_INDEX'] is True:
            create_file.write("X_RAY_SPEC_INDEX    %s\n"%(Decimal(repr(params[parameter_number])).quantize(SIXPLACES)))
            parameter_number += 1
        else:
            create_file.write("X_RAY_SPEC_INDEX    %s\n"%(self.Fiducial_Params['X_RAY_SPEC_INDEX']))

        if self.param_legend['TVIR_MIN'] is True:
            create_file.write("X_RAY_TVIR_MIN    %s\n"%(Decimal(repr(X_RAY_TVIR_MIN)).quantize(SIXPLACES)))
        else:
            create_file.write("X_RAY_TVIR_MIN    %s\n"%(self.Fiducial_Params['X_RAY_TVIR_MIN']))            

        create_file.write("X_RAY_TVIR_LB    %s\n"%(self.Fiducial_Params['X_RAY_TVIR_LB']))
        create_file.write("X_RAY_TVIR_UB    %s\n"%(self.Fiducial_Params['X_RAY_TVIR_UB']))

        #create_file.write("F_STAR    %s\n"%(self.Fiducial_Params['F_STAR']))

        create_file.write("N_RSD_STEPS    %s\n"%(self.Fiducial_Params['N_RSD_SUBCELLS']))
        create_file.write("LOS_direction    %s\n"%(self.Fiducial_Params['LOS_direction']))

        if self.IncludeLightCone is False: 
            for i in range(number_redshifts):
                create_file.write("CO-EVAL-Z    %s\n"%(AllRedshifts[i]))        

        create_file.close() 

        if self.FlagOptions['GENERATE_NEW_ICS'] is True:
            # A random number between 1 and 10^12 should be sufficient to randomise the ICs
            RandomSeed = np.random.uniform(low=1,high=1e12,size=1)

        # Now create the cosmology file associated with this walker.
        create_file = open("WalkerCosmology_%s.txt"%(StringArgument_other),"w")
        if self.FlagOptions['GENERATE_NEW_ICS'] is True:
            create_file.write("RANDOM_SEED    %s\n"%(RandomSeed[0]))
        else:
            create_file.write("RANDOM_SEED    %s\n"%(Decimal(repr(1.0)).quantize(SIXPLACES)))

        if self.param_legend['SIGMA_8'] is True:
            create_file.write("SIGMA_8    %s\n"%(Decimal(repr(params[parameter_number])).quantize(SIXPLACES)))
            parameter_number += 1
        else:
            create_file.write("SIGMA_8    %s\n"%(self.Fiducial_Params['SIGMA_8']))

        if self.param_legend['littleh'] is True:    
            create_file.write("hubble    %s\n"%(Decimal(repr(params[parameter_number])).quantize(SIXPLACES)))
            parameter_number += 1
        else:
            create_file.write("hubble    %s\n"%(self.Fiducial_Params['littleh']))

        if self.param_legend['OMEGA_M'] is True:
            create_file.write("Omega_M    %s\n"%(Decimal(repr(params[parameter_number])).quantize(SIXPLACES)))
            parameter_number += 1
        else:
            create_file.write("Omega_M    %s\n"%(self.Fiducial_Params['OMEGA_M']))

        if self.param_legend['OMEGA_M'] is True:
            create_file.write("Omega_L    %s\n"%(Decimal(repr(1. - params[parameter_number-1])).quantize(SIXPLACES)))
        else:
            create_file.write("Omega_L    %s\n"%(Decimal(repr(1. - float(self.Fiducial_Params['OMEGA_M']))).quantize(SIXPLACES)))

        if self.param_legend['OMEGA_b'] is True:
            create_file.write("Omega_b    %s\n"%(Decimal(repr(params[parameter_number])).quantize(SIXPLACES)))
            parameter_number += 1
        else:
            create_file.write("Omega_b    %s\n"%(self.Fiducial_Params['OMEGA_b']))

        if self.param_legend['NS'] is True:
            create_file.write("ns    %s\n"%(Decimal(repr(params[parameter_number])).quantize(SIXPLACES)))
            parameter_number += 1
        else:
            create_file.write("ns    %s\n"%(self.Fiducial_Params['NS']))

        create_file.close()

        if self.FlagOptions['LOG_LINEAR_K_SAMPLING'] is True:
            kSplineMin = np.log10(self.Foreground_cut)
            kSplineMax = np.log10(self.Shot_Noise_cut)
        else:
            kSplineMin = self.Foreground_cut
            kSplineMax = self.Shot_Noise_cut

        kSpline = np.zeros(self.NSplinePoints)

        for j in range(self.NSplinePoints):
            kSpline[j] = kSplineMin + (kSplineMax - kSplineMin)*float(j)/(self.NSplinePoints - 1)

        if self.FlagOptions['LOG_LINEAR_K_SAMPLING'] is True:
            kSpline = 10**( kSpline )

        counter = 0

        command = "./drive_21cmMC_streamlined %s"%(StringArgument)
        print command
        os.system(command)

        total_sum = 0
        poisson_sum = 0

        
        if self.FlagOptions['KEEP_GLOBAL_DATA'] is True:
            k_values_estimate = np.loadtxt('AveData_%s.txt'%(StringArgument_other), usecols=(0,))
            PS_values_estimate = np.loadtxt('AveData_%s.txt'%(StringArgument_other), usecols=(2,))

            if self.IncludeLightCone is False:
                k_values_estimate = k_values_estimate[::-1]
                PS_values_estimate = PS_values_estimate[::-1]

            # Converting the redshifts to frequencies for the interpolation (must be in increasing order, it is by default redshift which is decreasing)
            FrequencyValues_mock = np.zeros(len(self.k_values[0]))
            FrequencyValues_model = np.zeros(len(k_values_estimate))

            # Shouldn't need two, as they should be the same sampling. However, just done it for now
            for j in range(len(self.k_values[0])):
                FrequencyValues_mock[j] = ((2.99792e8)/(.2112*(1. + self.k_values[0][j])))/(1e6)

            for j in range(len(k_values_estimate)):    
                FrequencyValues_model[j] = ((2.99792e8)/(.2112*(1. + k_values_estimate[j])))/(1e6)

            splined_mock = interpolate.splrep(FrequencyValues_mock,self.PS_values[0],s=0)
            splined_model = interpolate.splrep(FrequencyValues_model,PS_values_estimate,s=0)

            FrequencyMin = self.Fiducial_Params['MIN_FREQ']
            FrequencyMax = self.Fiducial_Params['MAX_FREQ']

            if self.FlagOptions['USE_GS_FIXED_ERROR'] is True: 
                ErrorOnGlobal = self.Fiducial_Params['CONST_ERROR']
                Bandwidth = self.Fiducial_Params['BANDWIDTH']

                FrequencyBins = int(np.floor((FrequencyMax-FrequencyMin)/Bandwidth)) + 1

                for j in range(FrequencyBins):

                    FrequencyVal = FrequencyMin + Bandwidth*j        

                    MockPS_val = interpolate.splev(FrequencyVal,splined_mock,der=0)

                    ModelPS_val = interpolate.splev(FrequencyVal,splined_model,der=0)
                    
                    total_sum += np.square( (MockPS_val - ModelPS_val)/ErrorOnGlobal ) 

            else:

                for j in range(len(self.Error_k_values[0])):

                    FrequencyVal = ((2.99792e8)/(.2112*(1. + self.Error_k_values[0][j])))/(1e6)

                    if FrequencyVal >= FrequencyMin and FrequencyVal <= FrequencyMax:

                        MockPS_val = interpolate.splev(FrequencyVal,splined_mock,der=0)

                        ModelPS_val = interpolate.splev(FrequencyVal,splined_model,der=0)
                        
                        total_sum += np.square( (MockPS_val - ModelPS_val)/self.PS_Error[0][j] ) 

            # New in v1.4
            #if self.IncludeLF is True:
            if self.IncludeLF:
                # At the moment I just put the redshift list by hand, but this part should be modified.
                #NUM_OF_REDSHIFTS_FOR_LF = 4
                for iz in range(len(self.Redshifts_For_LF)):
                    # Exclude bright-end (Muv < -20) from Lumnosity function 
                    Muv_i = []
                    phi_i = []
                    error_i = []
                    j = 0 
                    while j < len(self.Muv_values[iz]):
                        if self.Muv_values[iz][j] > -20. and self.Muv_values[iz][j]!=0.:
                            Muv_i.append(self.Muv_values[iz][j])
                            phi_i.append(self.phi_values[iz][j])
                            error_i.append(self.phi_Error[iz][j])
                        j = j + 1  

                    Muv_values_estimate0 = np.loadtxt('LF_estimate_%s_%s.txt'%(StringArgument_other,self.Redshifts_For_LF[iz]), usecols=(0,))
                    log10phi_values_estimate0 = np.loadtxt('LF_estimate_%s_%s.txt'%(StringArgument_other,self.Redshifts_For_LF[iz]), usecols=(1,))
                    Muv_values_estimate = Muv_values_estimate0[::-1]
                    log10phi_values_estimate = log10phi_values_estimate0[::-1]

                    LF_criterion = 1 #LF_criteion == 0: skip this chain.
                    # check whether Muv does not increase monotonically with halo mass. if not interpolation is not possible.
                    i_check = 0
                    while i_check < len(Muv_values_estimate)-1:
                        if (Muv_values_estimate[i_check] > Muv_values_estimate[i_check+1]):
                            LF_criterion = 0
                            #print ("Found Muv list reversed\n")
                            break
                        i_check = i_check + 1

                    if (max(Muv_values_estimate) <= min(self.Muv_values[iz])) or (min(Muv_values_estimate) >= max(self.Muv_values[iz])):
                        LF_criterion = 0
                
                    if (LF_criterion == 0):
                        total_sum = total_sum + 10000000000.
                    else:
                        LFestimate_Spline = interpolate.splrep(Muv_values_estimate, log10phi_values_estimate,s=0)
                        for ii in range(len(Muv_i)):
                            Muv_i_val = Muv_i[ii]
                            log10phi_i_val = interpolate.splev(Muv_i_val,LFestimate_Spline,der=0)
                            #total_sum = total_sum + np.square(phi_i[ii] - 10**(log10phi_i_val)) / (np.square(error_i[ii]))
                            chi2_i = np.square(phi_i[ii] - 10**(log10phi_i_val)) / (np.square(error_i[ii]))
                            if (np.isinf(chi2_i)):
                                chi2_i = 100000.
                            total_sum = total_sum + chi2_i

        else:
            if self.IncludeLightCone is True:

                # For the light-cone version, the c-code creates a single textfile containing the filenames of each of the light-cone 21cm PS generated. This 
                # should be of equal or greater length than the number of mock observations added.

                LightconePSFilename = 'delTps_lightcone_filenames_%s.txt'%(StringArgument_other)
                filename = open('%s'%(LightconePSFilename), 'r') 
                LightconePS = [line.rstrip('\n') for line in filename]

                #nf_vals[0] = 'Walker_%s.txt'%(StringArgument_other)
                nf_vals[0] = 0.#'Walker_%s.txt'%(StringArgument_other)

                for i in range(len(self.Redshift)):                
                    k_values_estimate = np.loadtxt('%s'%(LightconePS[i]), usecols=(0,)) 
                    PS_values_estimate = np.loadtxt('%s'%(LightconePS[i]), usecols=(1,))
                    Poisson_error_estimate = np.loadtxt('%s'%(LightconePS[i]), usecols=(2,)) # Read possion errors

                    if self.FlagOptions['KEEP_ALL_DATA'] is True:
                        if not self.IncludeLF is 2:
                            if i == 0:
                                StoredStatisticalData.append(k_values_estimate)
                                StoredFileLayout.append("{%i}"%(i))

                                StoredStatisticalData_Error.append(k_values_estimate)
                                StoredFileLayout_Error.append("{%i}"%(i))

                            StoredStatisticalData.append(PS_values_estimate)
                            StoredFileLayout.append("{%i}"%(i+1))

                            StoredStatisticalData_Error.append(Poisson_error_estimate)
                            StoredFileLayout_Error.append("{%i}"%(i+1))


            else:

                for i in range(len(AllRedshifts)):                                
                    # Read in the neutral fraction and 21cm PS for this parameter set and redshift
                    nf_value = np.loadtxt('NeutralFraction_%s_%s.txt'%(StringArgument_other,AllRedshifts[i]), usecols=(0,))

                    nf_vals[i] = nf_value

                    # This only reading the data in from file, and then saving it to output
                    # Yes, I end up reading twice, but whatever...
                    # (I split it in the case that Redshifts_for_Prior was non-zero)
                    if not self.IncludeLF is 2:
                        k_values_estimate = np.loadtxt('delTps_estimate_%s_%s.txt'%(StringArgument_other,AllRedshifts[i]), usecols=(0,))
                        PS_values_estimate = np.loadtxt('delTps_estimate_%s_%s.txt'%(StringArgument_other,AllRedshifts[i]), usecols=(1,))
                        Poisson_error_estimate = np.loadtxt('delTps_estimate_%s_%s.txt'%(StringArgument_other,AllRedshifts[i]), usecols=(2,))

                        if self.FlagOptions['KEEP_ALL_DATA'] is True:
                            if i == 0:
                                StoredStatisticalData.append(k_values_estimate)
                                StoredFileLayout.append("{%i}"%(i))

                                StoredStatisticalData_Error.append(k_values_estimate)
                                StoredFileLayout_Error.append("{%i}"%(i))


                            StoredStatisticalData.append(PS_values_estimate)
                            StoredFileLayout.append("{%i}"%(i+1))

                            StoredStatisticalData_Error.append(Poisson_error_estimate)
                            StoredFileLayout_Error.append("{%i}"%(i+1))


#                nf_vals[len(AllRedshifts)] = 'Walker_%s.txt'%(StringArgument_other)
            nf_vals[len(AllRedshifts)] = '%s'%(Individual_ID)
            nf_vals[len(AllRedshifts)+1] = '%s'%(Individual_ID_2)

            # Note here that the usage of len(Redshift) uses the number of mock lightcone 21cm PS if IncludeLightCone was set to True.
            for i in range(len(self.Redshift)):                
                if self.IncludeLightCone is True:
                    k_values_estimate = np.loadtxt('%s'%(LightconePS[i]), usecols=(0,)) 
                    PS_values_estimate = np.loadtxt('%s'%(LightconePS[i]), usecols=(1,))
                    Poisson_error_estimate = np.loadtxt('%s'%(LightconePS[i]), usecols=(2,)) # Read possion errors
                elif not self.IncludeLF is 2:
                    # Read in the neutral fraction and 21cm PS for this parameter set and redshift
                    k_values_estimate = np.loadtxt('delTps_estimate_%s_%s.txt'%(StringArgument_other,self.Redshift[i]), usecols=(0,))

                    PS_values_estimate = np.loadtxt('delTps_estimate_%s_%s.txt'%(StringArgument_other,self.Redshift[i]), usecols=(1,))
                    Poisson_error_estimate = np.loadtxt('delTps_estimate_%s_%s.txt'%(StringArgument_other,self.Redshift[i]), usecols=(2,))

                if not self.IncludeLF is 2:
                    splined_mock = interpolate.splrep(self.k_values[i],np.log10(self.PS_values[i]),s=0)
                    splined_error = interpolate.splrep(self.Error_k_values[i],np.log10(self.PS_Error[i]),s=0)

                    splined_model = interpolate.splrep(k_values_estimate,np.log10(PS_values_estimate),s=0)
                    splined_model_poisson_err = interpolate.splrep(k_values_estimate,np.log10(Poisson_error_estimate),s=0)

                    # Interpolating the mock and error PS in log space
                    for j in range(self.NSplinePoints):

                        MockPS_val = 10**(interpolate.splev(kSpline[j],splined_mock,der=0))
                        ErrorPS_val = 10**(interpolate.splev(kSpline[j],splined_error,der=0))

                        
                        ModelPS_val = 10**(interpolate.splev(kSpline[j],splined_model,der=0))
                        ModelPE_val = 10**(interpolate.splev(kSpline[j],splined_model_poisson_err,der=0))

                        # Check if there are any nan values for the 21cm PS
                        # A nan value implies a IGM neutral fraction of zero, that is, reionisation has completed and thus no 21cm signal
                        # Set the value of the 21cm PS to zero. Which results in the largest available difference (i.e. if you expect a signal
                        # (i.e. non zero mock 21cm PS) but have no signal from the sampled model, then want a large difference for the 
                        # chi-squared likelihood).
                        if np.isnan(ModelPS_val) == True:
                            ModelPS_val = 0.0

                        if np.isnan(ModelPE_val) == True:
                            ModelPE_val = 0.0

                        if np.isnan(MockPS_val) == True:
                            MockPS_val = 0.0
                    
                        if self.IncludePS is True:
                         #total_sum += np.square((MockPS_val - ModelPS_val)/(np.sqrt(ErrorPS_val**2. + (self.ModUncert*ModelPS_val)**2.)))                 
                              total_sum += np.square((MockPS_val - ModelPS_val)/(np.sqrt(ErrorPS_val**2. + ModelPE_val**2)))                 
                   # print MockPS_val, ModelPS_val, total_sum

            if self.IncludeMorphologyHII is True or self.IncludePoissonBubHII is True or self.IncludeMorphologyHI is True or self.IncludePoissonBubHI is True:
                 
                 lim1 = -100
                 lim2 = 100
                 vol_lim = 10
                 for ii in range(len(self.Redshift)):

                    data = load_binary_data('delta_T_%s_z%s_%s_250Mpc'%(StringArgument_other,self.Redshift[ii], str(self.HII_DIMS)))
                    data = data.reshape((self.HII_DIMS, self.HII_DIMS, self.HII_DIMS), order='C')
                    data.shape = (self.HII_DIMS, self.HII_DIMS, self.HII_DIMS)
					
					
                    #newdata2 =  data < 1.
                    
                    if(nf_vals[ii] < 0.02) :
                         newdata2 = np.ones([self.HII_DIMS, self.HII_DIMS, self.HII_DIMS])
                         curnf = 0.
                    elif(nf_vals[ii] > 0.98):
                         newdata2 =  np.zeros([self.HII_DIMS, self.HII_DIMS, self.HII_DIMS])
                         curnf = 1.
                    else :
                         noise = t2c.noise_cube_coeval(self.HII_DIMS, float(self.Redshift[ii]), obs_time=1000, boxsize=self.MPC_DIMS, total_int_time=6., int_time=10., \
                         declination=-30., uv_map=self.uv_maps[ii], N_ant=512, verbose=False, fft_wrap=False)

                         output_dtheta  = (1+float(self.Redshift[ii]))*21e-5/2.
                         output_ang_res = output_dtheta*t2c.cm.z_to_cdist(float(self.Redshift[ii])) \
                                     * self.HII_DIMS/float(self.MPC_DIMS)

                         kernel = t2c.gauss_kernel(self.HII_DIMS, sigma=1., fwhm=output_ang_res)
                         kernelfft = np.fft.fft2(np.fft.ifftshift(kernel))
                         kernelfft[self.uv_maps[ii] == 0] = 0
                         uv_map = kernelfft
                    
                         data_dtb_smo = np.zeros((data.shape))

                         for i in range(self.HII_DIMS):
                              data_dtb_smo[:,:,i] =np.real(np.fft.ifft2(np.multiply(np.fft.fft2(data[:,:,i]+noise[:,:,i]), uv_map/np.max(uv_map))))

                         output_met1 = np.zeros(data_dtb_smo.shape)
                         kernel = t2c.tophat_kernel(self.HII_DIMS, output_ang_res)
                         for i in range(self.HII_DIMS):
                              output_met1[i,:,:] =t2c.smooth_with_kernel(data_dtb_smo[i,:,:], kernel)
                         img = output_met1.astype(float)
                         # print("Smoothing --- %s seconds ---\n\n" %(time.time() - start_time)  )

                         allth=[]
                         for i in range(self.HII_DIMS):
                              th = skimage.filters.threshold_triangle(img[:,:,i])
                              allth.append(th)
                         newdata2 = img < np.median(allth)
                              # print("Threshold --- %s seconds ---\n\n" %(time.time() - start_time))    

                         curnf = 1. - np.count_nonzero(newdata2.astype(float)) / float(self.HII_DIMS**3)
                    
                    hdu = fits.PrimaryHDU(newdata2.astype('uint8'))
                    hdul = fits.HDUList([hdu])
                    hdul.writeto('thr_delta_T_%s_z%s_%s_250Mpc.fits'%(StringArgument_other,self.Redshift[ii], str(self.HII_DIMS)))
                   # print("WRITE FITS --- %s seconds ---\n\n" %(time.time() - start_time)    )


                    if self.IncludeMorphologyHII is True or self.IncludePoissonBubHII is True:
                         command = "/Users/users/gazagnes/disccoman/./disccoman -g 1,1,1 -l 5 --intype fits  --attribute 4 -c 26  --threads 1   -f eor --eor hii  --bpp 8  \
                         --inprefix thr_delta_T_%s_z%s_%s_250Mpc --outprefix out_hii_%s_%s -v off "%(StringArgument_other,self.Redshift[ii],str(self.HII_DIMS),StringArgument_other,self.Redshift[ii])
                     #    command = "../../../disccoman/./disccoman -g 1,1,1 -l 5 --intype fits  --attribute 4 -c 26  --threads 1   -f eor --eor hii  --bpp 8  --inprefix thr_delta_T_%s_z%s_%s_%sMpc --outprefix out_hii_%s_%s -v off "%(StringArgument_other,self.Redshift[ii],str(self.HII_DIMS), str(self.MPC_DIMS),StringArgument_other,self.Redshift[ii])
                         os.system(command)
                       #  print("DISCCOMAN --- %s seconds ---\n\n" %(time.time() - start_time) )   

                         
                         name =  "out_hii_%s_%s"%(StringArgument_other,self.Redshift[ii])
                         f = open(name+'_vol.bin', "r")
                         volume = np.fromfile(f, dtype=np.uint64)*2

                         f = open(name+'_elong.bin', "r")
                         all_e = np.fromfile(f, dtype=np.float)

                         f = open(name+'_flat.bin', "r")
                         all_f = np.fromfile(f, dtype=np.float)
                    
                         f = open(name+'_spars.bin', "r")
                         all_s = np.fromfile(f, dtype=np.float)
                         
                         f = open(name+'_ncomp.bin', "r")
                         all_n = np.fromfile(f, dtype=np.float)
                         
                         good = volume > vol_lim
                         volume = volume[good]
                         all_e = all_e[good]
                         all_f = all_f[good]
                         all_s = all_s[good]
                         all_n = all_n[good]
                         
                         model_dis = zip(np.log10(volume), all_s, all_e, all_f)
                         seen = set()
                         uniq = []
                         for x in model_dis:
                              if x not in seen:
                                   uniq.append(x)
                                   seen.add(x)

                         model_dis=list(uniq)
                         model_dis.extend([[-1,lim1,lim1,lim1],[-1,lim2,lim1,lim1],[-1,lim1,lim1,lim2],[-1,lim1,lim2,lim1]])
                         model_dis.extend([[-1,lim2,lim2,lim1],[-1,lim2,lim2,lim2],[-1,lim1,lim2,lim2], [-1,lim2,lim1,lim2]])
                         model_dis.extend([[16,lim1,lim1,lim1],[16,lim2,lim1,lim1], [16,lim2,lim2,lim1],[16,lim2,lim2,lim2],[16,lim1,lim2,lim2], [16,lim1,lim1,lim2],[16,lim2,lim1,lim2],[16,lim1,lim2,lim1]])

                         model_vor = Voronoi(model_dis, qhull_options='QJ')
                         model_vol = voronoi_volumes(model_vor)
                      
                         voronoi_kdtree = cKDTree(self.mock_hii_dis[ii])
                         size_data = len(self.mock_hii_dis[ii][:])
                         size_model = len(model_dis[:])
    
                         totp = 0
                         nval = 0


                         """for i,p in enumerate(self.mock_hii_vor[ii].points):
                              vor_pt = p
                              test_point_dist, test_point_regions = voronoi_kdtree.query(vor_pt, k=1)
                              pvol = 1/(model_vol[test_point_regions]*size_model)#
                             # if model_vol[test_point_regions] >1e30:
								  #print "ERROR \n\n\n\n\n\n\n\n"
							#	  continue
                              nval+=1            

                              totp+=np.log(pvol)
                              
                         if self.IncludePoissonBubHII is True:
                              poisson = np.log(scipy.stats.poisson.pmf(size_data-16,size_model-16))
                         else:
                              poisson = 0
"""

                         for i in range(self.mock_hii_vor[ii].npoints):
                              pt =randrange(model_vor.npoints)
                              test_point_dist, test_point_regions = voronoi_kdtree.query(model_vor.points[pt], k=1)
                             # p2 = 1/(model_vol[pt]*size_model)
                             # p1 = 1/(self.mock_hii_vol[ii][test_point_regions]*size_data)
                             # lamb = p1/p2
                             # pvol = np.log(scipy.stats.poisson.pmf(1,lamb))
							  #  if(vol_2 > 1e20 or vol_1 > 1e20 ):
							  #      print "err"
                              #if pvol == float("-inf"):
							#		pvol = -100
                              totp+=test_point_dist#pvol

                              nval+=1
                              
                         if self.IncludePoissonBubHII is True:
                              poisson = np.log(scipy.stats.poisson.pmf(size_data-16,size_model-16))
                         else:
                              poisson = 0                                      
                         total_sum += -0.5*totp+poisson
                         if(np.isinf(total_sum)):
                              total_sum = -100000000
                              
                         print  (Decimal(repr(params[0])).quantize(SIXPLACES)), (Decimal(repr(params[1])).quantize(SIXPLACES)), (Decimal(repr(params[2])).quantize(SIXPLACES))
                         print "Number of values combined: %ld (size data %ld, Size model %ld)"%(nval, size_data, size_model)
                         print "Value of the combined Pi %f"%totp
                         print "Value of the Poisson distribution: %f"%poisson
                         print 
                                                              

                         command = "rm out_hii_%s_%s*"%(StringArgument_other,self.Redshift[ii])
                         os.system(command)
                         
                         #total_sum = total_sum / npa

                        # print("COMPUTED SUM --- %s seconds ---\n\n" %(time.time() - start_time))   
                    

                    if self.IncludeMorphologyHI is True or self.IncludePoissonBubHI is True:
                        # command = "/Users/users/gazagnes/disccoman/./disccoman -g 1,1,1 -l 5 --intype fits  --attribute 4 -c 26  --threads 1   -f eor --eor hi  --bpp 8  --inprefix thr_delta_T_%s_z%s_%s_%sMpc --outprefix out_hi_%s_%s -v off "%(StringArgument_other,self.Redshift[ii],str(self.HII_DIMS), str(self.MPC_DIMS), StringArgument_other,self.Redshift[ii])
                         #command = "../../../disccoman/./disccoman -g 1,1,1 -l 5 --intype fits  --attribute 4 -c 26  --threads 1   -f eor --eor hi  --bpp 8  --inprefix thr_delta_T_%s_z%s_%s_%sMpc --outprefix out_hi_%s_%s -v off "%(StringArgument_other,self.Redshift[ii],str(self.HII_DIMS), str(self.MPC_DIMS),StringArgument_other,self.Redshift[ii])
                         os.system(command)
                         name =  "out_hi_%s_%s"%(StringArgument_other,self.Redshift[ii])
                         f = open(name+'_vol.bin', "r")
                         volume = np.fromfile(f, dtype=np.uint64)

                         f = open(name+'_elong.bin', "r")
                         all_e = np.fromfile(f, dtype=np.float)
                         
                         f = open(name+'_flat.bin', "r")
                         all_f = np.fromfile(f, dtype=np.float)
                         
                         f = open(name+'_spars.bin', "r")
                         all_s = np.fromfile(f, dtype=np.float)
                    
                         f = open(name+'_ncomp.bin', "r")
                         all_n = np.fromfile(f, dtype=np.float)

                        
                        # print StringArgument_other, total_sum, poisson_sum
                         command = "rm out_hi_%s_%s*"%(StringArgument_other,self.Redshift[ii])
                         os.system(command)



               
               
            # New in v1.4
            #if self.IncludeLF is True:
            if self.IncludeLF:
                # At the moment I just put the redshift list by hand, but this part should be modified.
                #NUM_OF_REDSHIFTS_FOR_LF = 4
                for iz in range(len(self.Redshifts_For_LF)):
                    # Exclude bright-end (Muv < -20) from Lumnosity function 
                    Muv_i = []
                    phi_i = []
                    error_i = []
                    j = 0 
                    while j < len(self.Muv_values[iz]):
                        if self.Muv_values[iz][j] > -20. and self.Muv_values[iz][j]!=0.:
                            Muv_i.append(self.Muv_values[iz][j])
                            phi_i.append(self.phi_values[iz][j])
                            error_i.append(self.phi_Error[iz][j])
                        j = j + 1  

                    Muv_values_estimate0 = np.loadtxt('LF_estimate_%s_%s.txt'%(StringArgument_other,self.Redshifts_For_LF[iz]), usecols=(0,))
                    log10phi_values_estimate0 = np.loadtxt('LF_estimate_%s_%s.txt'%(StringArgument_other,self.Redshifts_For_LF[iz]), usecols=(1,))
                    Muv_values_estimate = Muv_values_estimate0[::-1]
                    log10phi_values_estimate = log10phi_values_estimate0[::-1]

                    LF_criterion = 1 #LF_criteion == 0: skip this chain.
                    # check whether Muv does not increase monotonically with halo mass. if not interpolation is not possible.
                    i_check = 0
                    while i_check < len(Muv_values_estimate)-1:
                        if (Muv_values_estimate[i_check] > Muv_values_estimate[i_check+1]):
                            LF_criterion = 0
                            #print ("Found Muv list reversed\n")
                            break
                        i_check = i_check + 1

                    if (max(Muv_values_estimate) <= min(self.Muv_values[iz])) or (min(Muv_values_estimate) >= max(self.Muv_values[iz])):
                        LF_criterion = 0
                
                    if (LF_criterion == 0):
                        total_sum = total_sum + 10000000000.
                    else:
                        LFestimate_Spline = interpolate.splrep(Muv_values_estimate, log10phi_values_estimate,s=0)
                        for ii in range(len(Muv_i)):
                            Muv_i_val = Muv_i[ii]
                            log10phi_i_val = interpolate.splev(Muv_i_val,LFestimate_Spline,der=0)
                            #total_sum = total_sum + np.square(phi_i[ii] - 10**(log10phi_i_val)) / (np.square(error_i[ii]))
                            chi2_i = np.square(phi_i[ii] - 10**(log10phi_i_val)) / (np.square(error_i[ii]))
                            if (np.isinf(chi2_i)):
                                chi2_i = 100000.
                            total_sum = total_sum + chi2_i


            if self.FlagOptions['KEEP_ALL_DATA'] is True:
                StoredFileLayout = string.join(StoredFileLayout,separator_column)
                StoredFileLayout_Error = string.join(StoredFileLayout_Error,separator_column)

                with open('%s/StatisticalData/TotalPSData_%s.txt'%(self.FlagOptions['KEEP_ALL_DATA_FILENAME'],StringArgument_other),'w') as f:            
                    for x in zip(*StoredStatisticalData):
                        f.write("%s\n"%(StoredFileLayout).format(*x))

                with open('%s/StatisticalData_Error/TotalPS_ErrorData_%s.txt'%(self.FlagOptions['KEEP_ALL_DATA_FILENAME'],StringArgument_other),'w') as f:            
                    for x in zip(*StoredStatisticalData_Error):
                        f.write("%s\n"%(StoredFileLayout_Error).format(*x))


                f.close()

        if (self.PriorLegend['PlanckPrior'] is True and number_redshifts > 2) or self.PriorLegend['McGreerPrior'] is True or self.PriorLegend['GreigPrior'] is True or self.FlagOptions['KEEP_ALL_DATA'] is True:

            z_Hist = np.loadtxt('AveData_%s.txt'%(StringArgument_other), usecols=(0,))
            xH_Hist = np.loadtxt('AveData_%s.txt'%(StringArgument_other), usecols=(1,))

            # When the light-cone version is set, the values are writted in decreasing order, not increasing order
            # Therefore, reverse to be in increasing order (the interpolation/extrapolation is required to be in increasing order)
            if self.IncludeLightCone is True:
                if z_Hist[0] > z_Hist[-1]:
                    z_Hist = z_Hist[::-1]
                    xH_Hist = xH_Hist[::-1]                            

        if (self.FlagOptions['KEEP_ALL_DATA'] is True or self.PriorLegend['PlanckPrior'] is True) and number_redshifts > 2:

            # Mean and one sigma errors for the Planck constraints
            # The Planck prior is modelled as a Gaussian: tau = 0.058 \pm 0.012 (https://arxiv.org/abs/1605.03507)
            PlanckTau_Mean = 0.058
            PlanckTau_OneSigma = 0.012

            # Simple linear extrapolation of the redshift range provided by the user, to be able to estimate the optical depth
            nZinterp = 15

            # The minimum of the extrapolation is chosen to 5.9, to correspond to the McGreer et al. prior on the IGM neutral fraction.
            # The maximum is chosed to be z = 18., which is arbitrary.
            ZExtrap_min = 5.9
            ZExtrap_max = 20.0

            ZExtrapVals = np.zeros(nZinterp)
            XHI_ExtrapVals = np.zeros(nZinterp)

            # Perform only a linear interpolation/extrapolation
            order = 1

            # The linear interpolation/extrapolation function, taking as input the redshifts supplied by the user and the corresponding neutral fractions
            # recovered for the specific EoR parameter set
            LinearInterpolationFunction = InterpolatedUnivariateSpline(z_Hist, xH_Hist, k=order)

            for i in range(nZinterp):
                ZExtrapVals[i] = ZExtrap_min + (ZExtrap_max - ZExtrap_min)*float(i)/(nZinterp - 1)
    
                XHI_ExtrapVals[i] = LinearInterpolationFunction(ZExtrapVals[i])
            
                # Ensure that the neutral fraction does not exceed unity, or go negative
                if XHI_ExtrapVals[i] > 1.0:
                    XHI_ExtrapVals[i] = 1.0
                if XHI_ExtrapVals[i] < 0.0:
                    XHI_ExtrapVals[i] = 0.0

            # Set up the arguments for calculating the estimate of the optical depth. Once again, performed using command line code.
            separator_Planck = " "
            seq_Planck = []
            for i in range(nZinterp):
                seq_Planck.append("%s"%(ZExtrapVals[i])) 
                seq_Planck.append("%s"%(XHI_ExtrapVals[i]))    

            StringArgument_Planck = string.join(seq_Planck,separator_Planck)

            # Perform the computation of tau
            command = './ComputingTau_e %s %s %s'%(Individual_ID,Decimal(repr(params[0])).quantize(SIXPLACES),StringArgument_Planck)
            os.system(command)

            # Read tau from file
            tau_value = np.loadtxt('Tau_e_%s_%s.txt'%(Individual_ID,Decimal(repr(params[0])).quantize(SIXPLACES)), usecols=(0,))
            # remove the temporary files
            if self.FlagOptions['KEEP_ALL_DATA'] is True:
                command = "mv Tau_e_%s_%s.txt %s/TauData/"%(Individual_ID,Decimal(repr(params[0])).quantize(SIXPLACES),self.FlagOptions['KEEP_ALL_DATA_FILENAME'])
            else:
                command = "rm Tau_e_%s_%s.txt"%(Individual_ID,Decimal(repr(params[0])).quantize(SIXPLACES))
            
            os.system(command)

            # As the likelihood is computed in log space, the addition of the prior is added linearly to the existing chi^2 likelihood
            if self.PriorLegend['PlanckPrior'] is True:
                total_sum = total_sum + np.square( ( PlanckTau_Mean - tau_value )/(PlanckTau_OneSigma) )

            #if self.IncludeLightCone is True:
            #    nf_vals[1] = tau_value
            #else:
            #    # it is len(AllRedshifts) as the indexing begins at zero
            #    nf_vals[len(AllRedshifts)+2] = tau_value
            nf_vals[len(AllRedshifts)+2] = tau_value

        if self.PriorLegend['McGreerPrior'] is True:

            # Mean and one sigma errors for the McGreer et al. constraints
            # Modelled as a flat, unity prior at x_HI <= 0.06, and a one sided Gaussian at x_HI > 0.06 ( Gaussian of mean 0.06 and one sigma of 0.05 )
            McGreer_Mean = 0.06
            McGreer_OneSigma = 0.05            

            if McGreer_Redshift in z_Hist:

                for i in range(len(z_Hist)):
                    if z_Hist[i] == McGreer_Redshift:                        
                        McGreer_NF = xH_Hist[i]

                if McGreer_NF > 1.:
                    McGreer_NF = 1.
                if McGreer_NF < 0.:
                    McGreer_NF = 0.

                # As the likelihood is computed in log space, the addition of the prior is added linearly to the existing chi^2 likelihood
                if McGreer_NF <= 0.06:
                    total_sum = total_sum + 0.0 # Add zero, as we assume flat (unity) probability at x_HI <= 0.06 (as it is a lower limit)
                else:
                    total_sum = total_sum + np.square( ( McGreer_Mean - McGreer_NF )/(McGreer_OneSigma) )


            elif number_redshifts > 2:

                # Perform only a linear interpolation/extrapolation
                order = 1

                # The linear interpolation/extrapolation function, taking as input the redshifts supplied by the user and the corresponding neutral fractions
                # recovered for the specific EoR parameter set
                LinearInterpolationFunction = InterpolatedUnivariateSpline(z_Hist, xH_Hist, k=order)

                McGreer_NF = LinearInterpolationFunction(McGreer_Redshift)

                if McGreer_NF > 1.:
                    McGreer_NF = 1.
                if McGreer_NF < 0.:
                    McGreer_NF = 0.

                # As the likelihood is computed in log space, the addition of the prior is added linearly to the existing chi^2 likelihood
                if McGreer_NF <= 0.06:
                    total_sum = total_sum + 0.0 # Add zero, as we assume flat (unity) probability at x_HI <= 0.06 (as it is a lower limit)
                else:
                    total_sum = total_sum + np.square( ( McGreer_Mean - McGreer_NF )/(McGreer_OneSigma) )

        if self.PriorLegend['GreigPrior'] is True:

            # Interpolate the QSO damping wing PDF
            spline_QSODampingPDF = interpolate.splrep(self.NFValsQSO,self.PDFValsQSO,s=0)

            if QSO_Redshift in z_Hist:

                for i in range(len(z_Hist)):
                    if z_Hist[i] == QSO_Redshift:                        
                        NF_QSO = xH_Hist[i]

                # Ensure that the neutral fraction does not exceed unity, or go negative
                if NF_QSO > 1.0:
                    NF_QSO = 1.0
                if NF_QSO < 0.0:
                    NF_QSO = 0.0

                QSO_Prob = interpolate.splev(NF_QSO,spline_QSODampingPDF,der=0)

                # Interpolating the PDF from the QSO damping wing might cause small negative values at the edges (i.e. x_HI ~ 0 or ~1)
                # In case it is zero, or negative, set it to a very small non zero number (we take the log of this value, it cannot be zero)
                if QSO_Prob <= 0.0:
                    QSO_Prob = 0.000006

                # We work with the log-likelihood, therefore convert the IGM Damping wing PDF to log space
                QSO_Prob = -2.*np.log(QSO_Prob)

                total_sum = total_sum + QSO_Prob

            elif number_redshifts > 2:
            
                order = 1

                # Check the redshift range input by the user to determine whether to interpolate or extrapolate the IGM neutral fraction to the QSO redshift
                if QSO_Redshift < np.amin(self.Redshift):
                    # The QSO redshift is outside the range set by the user. Need to extrapolate the reionisation history to obtain the neutral fraction at the QSO redshift

                    # The linear interpolation/extrapolation function, taking as input the redshifts supplied by the user and the corresponding neutral fractions
                    # recovered for the specific EoR parameter set
                    LinearInterpolationFunction = InterpolatedUnivariateSpline(self.Redshift, nf_vals, k=order)

                    NF_QSO = LinearInterpolationFunction(QSO_Redshift)
                            
                else:
                    # The QSO redshift is within the range set by the user. Can interpolate the reionisation history to obtain the neutral fraction at the QSO redshift

                    spline_reionisationhistory = interpolate.splrep(self.Redshift,nf_vals,s=0)

                    NF_QSO = interpolate.splev(QSO_Redshift,spline_reionisationhistory,der=0)

                # Ensure that the neutral fraction does not exceed unity, or go negative
                if NF_QSO > 1.0:
                    NF_QSO = 1.0
                if NF_QSO < 0.0:
                    NF_QSO = 0.0

                QSO_Prob = interpolate.splev(NF_QSO,spline_QSODampingPDF,der=0)

                # Interpolating the PDF from the QSO damping wing might cause small negative values at the edges (i.e. x_HI ~ 0 or ~1)
                # In case it is zero, or negative, set it to a very small non zero number (we take the log of this value, it cannot be zero)
                if QSO_Prob <= 0.0:
                    QSO_Prob = 0.000006

                # We work with the log-likelihood, therefore convert the IGM Damping wing PDF to log space
                QSO_Prob = -2.*np.log(QSO_Prob)

                total_sum = total_sum + QSO_Prob
        
        if self.IncludeLightCone is True:

            if self.FlagOptions['KEEP_GLOBAL_DATA'] is True:

                LightconePSFilename = 'delTps_lightcone_filenames_%s.txt'%(StringArgument_other)
                filename = open('%s'%(LightconePSFilename), 'r') 
                LightconePS = [line.rstrip('\n') for line in filename]

            if self.FlagOptions['KEEP_ALL_DATA'] is True:
                command = "mv %s %s/StatisticalData/"%(LightconePSFilename,self.FlagOptions['KEEP_ALL_DATA_FILENAME'])                
            else:
                command = "rm %s"%(LightconePSFilename)
            
            os.system(command)

            # Removal of the individual light cone files is done here as in principle these can exceed the number of mock observations provided
            for i in range(len(LightconePS)):
                command = "rm %s"%(LightconePS[i])
                os.system(command)

            if self.FlagOptions['KEEP_ALL_DATA'] is True:
                for j in range(len(self.Redshifts_For_LF)):
                    command = "mv LF_estimate_%s_%s.txt %s/LFData/"%(StringArgument_other,self.Redshifts_For_LF[j],self.FlagOptions['KEEP_ALL_DATA_FILENAME'])
                    os.system(command)
            else:
                for j in range(len(self.Redshifts_For_LF)):
                    command = "rm LF_estimate_%s_%s.txt"%(StringArgument_other,self.Redshifts_For_LF[j])
                    os.system(command)
        else:
            
            if not self.IncludeLF is 2:
                command = "rm delTps_estimate_%s_*"%(StringArgument_other)
                os.system(command)

            command = "rm NeutralFraction_%s_*"%(StringArgument_other)
            os.system(command)

            if self.FlagOptions['KEEP_ALL_DATA'] is True:
                for j in range(len(self.Redshifts_For_LF)):
                    command = "mv LF_estimate_%s_%s.txt %s/LFData/"%(StringArgument_other,self.Redshifts_For_LF[j],self.FlagOptions['KEEP_ALL_DATA_FILENAME'])
                os.system(command)
            else:
                for j in range(len(self.Redshifts_For_LF)):
                    command = "rm LF_estimate_%s_%s.txt"%(StringArgument_other,self.Redshifts_For_LF[j])
                os.system(command)

        if OutputGlobalAve == 1:
            if self.FlagOptions['KEEP_ALL_DATA'] is True:
                command = "mv AveData_%s.txt %s/AveData/"%(StringArgument_other,self.FlagOptions['KEEP_ALL_DATA_FILENAME'])
            else:
                command = "rm AveData_%s.txt"%(StringArgument_other)
            
            os.system(command)

            

        if self.FlagOptions['KEEP_ALL_DATA'] is True:
            command = "mv Walker_%s.txt %s/WalkerData"%(StringArgument_other,self.FlagOptions['KEEP_ALL_DATA_FILENAME'])
            os.system(command) 

            command = "mv WalkerCosmology_%s.txt %s/WalkerData"%(StringArgument_other,self.FlagOptions['KEEP_ALL_DATA_FILENAME'])
            os.system(command) 
        else:
            command = "rm Walker_%s.txt"%(StringArgument_other)
            os.system(command) 

            command = "rm WalkerCosmology_%s.txt"%(StringArgument_other)
            os.system(command) 
            
        command = "rm *delta_T_%s*"%(StringArgument_other)
        os.system(command)
          
      #  if(np.isinf(total_sum)):
      #      total_sum = 10000000000.
        print  (Decimal(repr(params[0])).quantize(SIXPLACES)), (Decimal(repr(params[1])).quantize(SIXPLACES)), (Decimal(repr(params[2])).quantize(SIXPLACES))
        print "Final Likelihood: %f"%total_sum
        return total_sum,nf_vals

    def computeLikelihood(self, ctx):

        return self.Likelihood(ctx)

    def setup(self):
        print "Likelihood Fitting for 21cm Fast" 
