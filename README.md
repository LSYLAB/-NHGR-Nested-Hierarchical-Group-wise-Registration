## **Nested Hierarchical Group-wise Registration with a Graph-based Subgrouping Strategy for Efficient Template Construction （NHGR）**

Accurate and efficient group-wise registration for medical images is fundamentally important to construct a common template image for population-level analysis. However, current group-wise registration faces the challenges posed by the algorithm’s efficiency and capacity, and adaptability to large variations in the subject populations. This paper addresses these challenges with a novel Nested Hierarchical Group-wise Registration (NHGR) framework. Firstly, to alleviate the registration burden due to significant population variations, a new subgrouping strategy is proposed to serve as a “divide and conquer” mechanism that divides a large population into smaller subgroups. The subgroups with a hierarchical sequence are formed by gradually expanding the scale factors that relate to feature similarity and then conducting registration at the subgroup scale as the multi-scale conquer strategy. Secondly, the nested hierarchical group-wise registration is proposed to conquer the challenges due to the efficiency and capacity of the model from three perspectives. (1) Population level: the global group-wise registration is performed to generate age-related sub-templates from local subgroups progressively to the global population. (2) Subgroup level: the local group-wise registration is performed based on local image distributions to reduce registration error and achieve rapid optimization of sub-templates. (3) Image pair level: a deep multi-resolution registration network is employed for better registration efficiency. The proposed framework was evaluated on the brain datasets of adults and adolescents, respectively from 18 to 96 years and 5 to 21 years. Experimental results consistently demonstrated that our proposed group-wise registration method achieved better performance in terms of registration efficiency, template sharpness, and template centrality.

### **Prerequisites**
Python 3.5.2+; 

Pytorch 1.3.0 - 1.7.0;

NumPy; 

NiBabel

This code has been tested with Pytorch 1.6.0 and GTX1080TI GPU.

### **Code implementation:**
Step1: One low-frequency image and seven high-resolution images are obtained by 3D wavelet transform of the pre-processed data. (3Dwavlet.py)

Step2: Energy maps are obtained by seven high-resolution images. (EnergyMap.py)

Step3: Pretaining feature extraction network. (Train_VAE_age_ED.py)

Step4: Subgrouping based on persistent homology. (distribution_final.ipynb)

Step5: Group-wise registration is performed from local subgroups to global population. (Train_WAT1-6_mix.py)

### **Note:**
When the article is published, we will further refine the ReadMe and code.Thanks for understanding!
