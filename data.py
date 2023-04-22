'''
From the results, we can see that the accuracy of the single classifier is 66.67%. The accuracy of the ensemble classifiers is higher than the single classifier, with the ensemble classifier with 100 models achieving the highest accuracy of 80.00%. The accuracy of the ensemble classifier improves as the number of models increases. This is because bagging reduces the variance of the model by averaging the predictions of multiple models, resulting in a more stable and accurate ensemble classifier.
'''
'''
The accuracy of the single classifier and the ensemble classifiers with 10, 50, and 100 classifiers are all the same, which suggests that using bagging did not improve the classification accuracy for this specific dataset and classifier. This could be due to a variety of reasons, such as the dataset being too small or the classifier already performing at its maximum potential. Additionally, it is important to note that bagging may not always improve classification accuracy and should be used with caution and careful evaluation.
'''

data_set = [[0.17020707787494, 0.15, 0.49972782534513,
            1.9937239478858, "Plastic"],
            [0.13069872711354, 0.14134544193659, 0.75, 2.3108048761799, "Ceramic"],
            [0.13231035681923, 0.09832834720894, 0.75, 4.0376298951106, "Ceramic"],
            [0.065664632481769, 0.067778130408361,
             0.2610835190475, 3.2572147885865, "Ceramic"],
            [0.08822035725227, 0.10375412229737,
             0.58693755188256, 5.9682472309767, "Ceramic"],
            [0.11390475049809, 0.063194257766677,
             0.29646919535004, 4.7685331467876, "Metal"],
            [0.092248654685935, 0.083395750249511,
             0.34107086124483, 1.5838727549496, "Metal"],
            [0.07103272594019, 0.03, 0.11611723614407, 3.4696077311462, "Metal"],
            [0.094855433393443, 0.054799013619216,
             0.2054467202965, 3.7551109842814, "Metal"],
            [0.10719966345502, 0.096441079365225,
             0.63367867012613, 4.2331845715187, "Ceramic"],
            [0.05770781853263, 0.03, 0.13092558863574, 3.9202925262721, "Metal"],
            [0.11309448698099, 0.12026864776031, 0.75, 2.1870429735041, "Ceramic"],
            [0.06565184784827, 0.086184234958369,
             0.23144543453549, 2.6888377127345, "Metal"],
            [0.088343998360058, 0.03, 0.22026475788374, 2.0853142966468, "Ceramic"],
            [0.067433604421316, 0.034837284430762,
             0.12601151729064, 0.17416337315017, "Metal"],
            [0.14673583689539, 0.10569865938449,
             0.41358825740895, 1.9077428928291, "Plastic"],
            [0.081627264480676, 0.039114675393891,
             0.23954223970352, 2.0119286350506, "Ceramic"],
            [0.090556084589423, 0.087688843707091,
             0.3510448008883, 2.6829524056228, "Metal"],
            [0.1674726036475, 0.15, 0.56571823724904, 2.8184679776635, "Plastic"],
            [0.087461903630758, 0.13316749478598,
             0.70492996831831, 2.6323890421371, "Ceramic"],
            [0.056336427022749, 0.035315342802002,
             0.16390438674562, 3.8065136273554, "Ceramic"],
            [0.16189963198768, 0.11309886083338,
             0.34741633843916, 1.9271485661175, "Plastic"],
            [0.10293661587004, 0.088496392043282,
             0.6434817926885, 4.3507758321598, "Ceramic"],
            [0.15679782642034, 0.130538167537,
                0.49659025896301, 1.947565958217, "Plastic"],
            [0.1371556503088, 0.099223680948666, 0.75, 6.2831853071796, "Ceramic"],
            [0.095679023614052, 0.067296719342352,
             0.14471381049573, 3.8024440505698, "Plastic"],
            [0.10677868530699, 0.064663593821291,
             0.16517404541225, 4.1657610437738, "Plastic"],
            [0.076865055835569, 0.039407564846317,
             0.18577720107632, 4.4423824095994, "Metal"],
            [0.12729547945912, 0.10209332017592,
             0.36936322741347, 1.4663687684804, "Plastic"],
            [0.0922305122042, 0.077875169108831,
                0.31818831301151, 3.9747052876475, "Metal"],
            [0.15332145670679, 0.15, 0.50311484914552, 3.9003509143055, "Plastic"],
            [0.10937948027747, 0.097024256749921,
             0.28569678711778, 2.2020243883025, "Plastic"],
            [0.10087902430079, 0.044911668921174,
             0.30301199476658, 1.4360383275783, "Ceramic"],
            [0.1246859041714, 0.13034622185617,
                0.30398215403095, 3.7292179251944, "Plastic"],
            [0.053016541453632, 0.050798185869094,
             0.16978432395103, 5.706539406012, "Metal"],
            [0.084710817939262, 0.062168957354975,
             0.25305165623095, 3.1855660533865, "Metal"],
            [0.063001427535875, 0.053512580863868,
             0.27483865403259, 3.462153471388, "Ceramic"],
            [0.090012632839949, 0.094505437219087,
             0.53115529550216, 4.9240204599954, "Ceramic"],
            [0.1093580332405, 0.075628282233772,
             0.53070776140062, 3.2784702209921, "Ceramic"],
            [0.13511028521501, 0.088897521259258,
             0.27241265827825, 2.9198020492539, "Plastic"],
            [0.14003636925409, 0.14227278016029,
             0.46494238311267, 4.6365237179896, "Plastic"],
            [0.12029250110079, 0.12368931641352,
                0.33128742688871, 4.23164247619, "Plastic"],
            [0.090942522926329, 0.06616584871505,
             0.12298994617065, 5.2012768375545, "Plastic"],
            [0.059019347530206, 0.085833828163942,
             0.25721435568767, 2.9798406627175, "Metal"],
            [0.15298111375335, 0.11554753439699,
             0.43242449977078, 3.5296701683539, "Plastic"],
            [0.12885393186954, 0.14231382162052,
             0.43351664050863, 4.0618775011599, "Plastic"],
            [0.12882906255867, 0.15, 0.45644171468841, 2.6610388697349, "Plastic"],
            [0.10215465576603, 0.11900501515547,
             0.74377435056441, 2.4256489437523, "Ceramic"],
            [0.086150607671354, 0.095336338935588,
             0.53187685373455, 4.7658979694189, "Ceramic"],
            [0.089629241227652, 0.15, 0.27792063815096, 3.315270902435, "Plastic"],
            [0.12834894290456, 0.10493996660653, 0.75, 4.3674975310879, "Ceramic"],
            [0.13990827929484, 0.14892196872845,
             0.47962591909821, 1.6238026267868, "Plastic"],
            [0.061208447252443, 0.073691229269372,
             0.35335355941947, 2.7471014503591, "Ceramic"],
            [0.10359322619386, 0.086927507222009,
             0.58206146451215, 2.9677966985078, "Ceramic"],
            [0.085967452407859, 0.074723906980999,
             0.34098819945385, 3.3226762621524, "Metal"],
            [0.15789892043026, 0.10911490890347, 0.75, 4.2331842416437, "Ceramic"],
            [0.10811148767608, 0.073073354113978,
             0.27019618120982, 2.5073731683685, "Plastic"],
            [0.092550608750899, 0.093747462843227,
             0.37232708383006, 4.4062911952273, "Metal"],
            [0.12433492535852, 0.094317079519777,
             0.23833355316548, 3.3363753829239, "Plastic"],
            [0.12521906040439, 0.03, 0.21516945776157, 2.4508433350573, "Ceramic"],
            [0.11421019333548, 0.061499011497294,
                0.41946439392279, 1.2323108869, "Ceramic"],
            [0.086305841200261, 0.088706042551795,
             0.13813602191822, 5.1599254543833, "Plastic"],
            [0.13173942942765, 0.12350256134312, 0.75, 3.9693903166633, "Ceramic"],
            [0.17614085214159, 0.15, 0.59507090665931, 3.2770072073404, "Plastic"],
            [0.13883939826687, 0.14586950873522,
             0.48208051512176, 3.6271201165173, "Plastic"],
            [0.087114450179299, 0.040985602960244,
             0.17124264932725, 3.0415779451434, "Metal"],
            [0.097205554354253, 0.12082764356137,
             0.44722862900288, 1.9448918765068, "Metal"],
            [0.093995758725724, 0.081373801927315,
             0.17820563154008, 4.5675306817912, "Plastic"],
            [0.069894299327156, 0.057475808459266,
             0.18097043030704, 4.4720946857941, "Metal"],
            [0.13219179363809, 0.1462469100281, 0.75, 2.4374534999852, "Ceramic"],
            [0.072342723460166, 0.052900744239827,
             0.20380004771085, 4.4623488025485, "Metal"],
            [0.063212105996395, 0.068636263927473,
             0.18594543828779, 2.7382468597687, "Metal"],
            [0.08054466586271, 0.078760972875113,
             0.28711471910625, 3.7590582021091, "Metal"],
            [0.074690367660323, 0.052169725959116,
             0.17030452801699, 3.0980124893767, "Metal"],
            [0.14321524616663, 0.093042848006309,
                0.75, 2.9057971012588, "Ceramic"],
            [0.054788576973847, 0.03, 0.1, 3.1698072854378, "Ceramic"],
            [0.10534422720185, 0.031797235495378,
             0.23495974754319, 5.0252000973009, "Ceramic"],
            [0.099065915948221, 0.064858514908295,
             0.27187505412172, 3.0025506701765, "Metal"],
            [0.088705744000464, 0.10275058003627,
             0.38073160010157, 3.0974832140969, "Metal"],
            [0.16402676447595, 0.13313608680274,
                0.51319890204588, 1.166915082904, "Plastic"],
            [0.13034428253682, 0.1431894187626, 0.75, 3.3550255296817, "Ceramic"],
            [0.10294234234813, 0.049744483601254,
             0.28890534337916, 3.5696918435571, "Ceramic"],
            [0.12571635127086, 0.096866960205948,
             0.29499479659882, 2.5411201532223, "Plastic"],
            [0.1000787818189, 0.049079912062959,
             0.38334020094691, 2.7126157787229, "Ceramic"],
            [0.1479167634974, 0.09629694280608,
                0.37493989282969, 3.0165807947013, "Plastic"],
            [0.070095038561427, 0.082803015683009,
             0.27997680945402, 2.5095168950956, "Metal"],
            [0.071539799029628, 0.099094488419242,
             0.30607331637046, 5.6755188917474, "Metal"],
            [0.088314676711907, 0.055753717297801,
             0.23851770774032, 2.5351990660329, "Metal"],
            [0.10117743397404, 0.078462535737341,
             0.5463657535781, 2.8197410994355, "Ceramic"],
            [0.16010494152036, 0.088232264042623,
             0.27486269882784, 4.1664651655491, "Plastic"],
            [0.059392814616617, 0.056644161744612,
             0.16617306287465, 3.1515905980752, "Metal"],
            [0.13732802832017, 0.11641313529938,
             0.35583122751175, 3.2965106038064, "Plastic"],
            [0.098794456448179, 0.07293995056557,
             0.44653160377371, 3.9966000858361, "Ceramic"],
            [0.13144038236467, 0.11243464170412,
                0.36673401347558, 2.107709010638, "Plastic"],
            [0.063882605913494, 0.039141483446014,
             0.15876934389026, 2.214379480064, "Metal"],
            [0.056980628195313, 0.03, 0.1, 3.8471510758446, "Metal"],
            [0.17448335708525, 0.12795995697385, 0.75, 3.4769681354175, "Ceramic"],
            [0.065051974960521, 0.067859242806417,
             0.22576054967965, 3.0937351722307, "Metal"],
            [0.079269329218908, 0.06688251911803,
             0.17133524483365, 3.4123526221123, "Metal"],
            [0.15690107689746, 0.11365590621391,
             0.41199747126945, 3.2710045559701, "Plastic"],
            [0.097193631972382, 0.11938458856116,
             0.32281275877487, 3.7375620092324, "Plastic"],
            [0.1423527128933, 0.13287156977883, 0.75, 4.6489844501143, "Ceramic"],
            [0.14423993136539, 0.1418123779467,
                0.40585325682191, 3.0284932210318, "Plastic"],
            [0.13077543955667, 0.11436096124224,
             0.38825897831714, 4.3441325277497, "Plastic"],
            [0.13529414917172, 0.10079306146734,
             0.41220662317937, 5.3875689912937, "Plastic"],
            [0.08029077693591, 0.089949956755307,
             0.23109198497505, 0.22955086284879, "Plastic"],
            [0.07, 0.03, 0.1263613926807, 4.1251527025521, "Plastic"],
            [0.10731236182375, 0.091328413129029,
             0.20150572191532, 3.287683732231, "Plastic"],
            [0.14016526264168, 0.15, 0.47820025528309, 2.4881620862859, "Plastic"],
            [0.16355608355763, 0.15, 0.59541779754136, 3.5940701708032, "Plastic"],
            [0.11430812759503, 0.13026803130455,
             0.37454041986459, 4.0690235355497, "Plastic"],
            [0.15580490224717, 0.12922673634604,
                0.455438495607, 1.0766524615684, "Plastic"],
            [0.1303665158626, 0.13898728203977,
                0.39856732851642, 2.974130602917, "Plastic"],
            [0.14079472913793, 0.11270116042956,
             0.37014175878874, 5.0658012970135, "Plastic"],
            [0.12170390878788, 0.15, 0.43637546134608, 4.5849465309494, "Plastic"],
            [0.12512354072637, 0.098538030610358,
             0.32976902323333, 4.9328468908403, "Plastic"],
            [0.1363492787987, 0.15, 0.48806204193694, 3.2134639574355, "Plastic"],
            [0.11850434159965, 0.11151623424643,
                0.34486100219079, 4.70666048632, "Plastic"],
            [0.084024664512799, 0.04972427698517,
             0.12449491368036, 5.2222544100339, "Plastic"],
            [0.099200676539949, 0.098970851204098, 0.28219507178034, 5.2644216179615, "Plastic"]]

trim_data = [[0.17020707787494, 0.15, 0.49972782534513, 1.9937239478858], [0.13069872711354, 0.14134544193659, 0.75, 2.3108048761799], [0.13231035681923, 0.09832834720894, 0.75, 4.0376298951106], [0.065664632481769, 0.067778130408361, 0.2610835190475, 3.2572147885865], [0.08822035725227, 0.10375412229737, 0.58693755188256, 5.9682472309767], [0.11390475049809, 0.063194257766677, 0.29646919535004, 4.7685331467876], [0.092248654685935, 0.083395750249511, 0.34107086124483, 1.5838727549496], [0.07103272594019, 0.03, 0.11611723614407, 3.4696077311462], [0.094855433393443, 0.054799013619216, 0.2054467202965, 3.7551109842814], [0.10719966345502, 0.096441079365225, 0.63367867012613, 4.2331845715187], [0.05770781853263, 0.03, 0.13092558863574, 3.9202925262721], [0.11309448698099, 0.12026864776031, 0.75, 2.1870429735041], [0.06565184784827, 0.086184234958369, 0.23144543453549, 2.6888377127345], [0.088343998360058, 0.03, 0.22026475788374, 2.0853142966468], [0.067433604421316, 0.034837284430762, 0.12601151729064, 0.17416337315017], [0.14673583689539, 0.10569865938449, 0.41358825740895, 1.9077428928291], [0.081627264480676, 0.039114675393891, 0.23954223970352, 2.0119286350506], [0.090556084589423, 0.087688843707091, 0.3510448008883, 2.6829524056228], [0.1674726036475, 0.15, 0.56571823724904, 2.8184679776635], [0.087461903630758, 0.13316749478598, 0.70492996831831, 2.6323890421371], [0.056336427022749, 0.035315342802002, 0.16390438674562, 3.8065136273554], [0.16189963198768, 0.11309886083338, 0.34741633843916, 1.9271485661175], [0.10293661587004, 0.088496392043282, 0.6434817926885, 4.3507758321598], [0.15679782642034, 0.130538167537, 0.49659025896301, 1.947565958217], [0.1371556503088, 0.099223680948666, 0.75, 6.2831853071796], [0.095679023614052, 0.067296719342352, 0.14471381049573, 3.8024440505698], [0.10677868530699, 0.064663593821291, 0.16517404541225, 4.1657610437738], [0.076865055835569, 0.039407564846317, 0.18577720107632, 4.4423824095994], [0.12729547945912, 0.10209332017592, 0.36936322741347, 1.4663687684804], [0.0922305122042, 0.077875169108831, 0.31818831301151, 3.9747052876475], [0.15332145670679, 0.15, 0.50311484914552, 3.9003509143055], [0.10937948027747, 0.097024256749921, 0.28569678711778, 2.2020243883025], [0.10087902430079, 0.044911668921174, 0.30301199476658, 1.4360383275783], [0.1246859041714, 0.13034622185617, 0.30398215403095, 3.7292179251944], [0.053016541453632, 0.050798185869094, 0.16978432395103, 5.706539406012], [0.084710817939262, 0.062168957354975, 0.25305165623095, 3.1855660533865], [0.063001427535875, 0.053512580863868, 0.27483865403259, 3.462153471388], [0.090012632839949, 0.094505437219087, 0.53115529550216, 4.9240204599954], [0.1093580332405, 0.075628282233772, 0.53070776140062, 3.2784702209921], [0.13511028521501, 0.088897521259258, 0.27241265827825, 2.9198020492539], [0.14003636925409, 0.14227278016029, 0.46494238311267, 4.6365237179896], [0.12029250110079, 0.12368931641352, 0.33128742688871, 4.23164247619], [0.090942522926329, 0.06616584871505, 0.12298994617065, 5.2012768375545], [0.059019347530206, 0.085833828163942, 0.25721435568767, 2.9798406627175], [0.15298111375335, 0.11554753439699, 0.43242449977078, 3.5296701683539], [0.12885393186954, 0.14231382162052, 0.43351664050863, 4.0618775011599], [0.12882906255867, 0.15, 0.45644171468841, 2.6610388697349], [0.10215465576603, 0.11900501515547, 0.74377435056441, 2.4256489437523], [0.086150607671354, 0.095336338935588, 0.53187685373455, 4.7658979694189], [0.089629241227652, 0.15, 0.27792063815096, 3.315270902435], [0.12834894290456, 0.10493996660653, 0.75, 4.3674975310879], [0.13990827929484, 0.14892196872845, 0.47962591909821, 1.6238026267868], [0.061208447252443, 0.073691229269372, 0.35335355941947, 2.7471014503591], [0.10359322619386, 0.086927507222009, 0.58206146451215, 2.9677966985078], [0.085967452407859, 0.074723906980999, 0.34098819945385, 3.3226762621524], [0.15789892043026, 0.10911490890347, 0.75, 4.2331842416437], [0.10811148767608, 0.073073354113978, 0.27019618120982, 2.5073731683685], [0.092550608750899, 0.093747462843227, 0.37232708383006, 4.4062911952273], [0.12433492535852, 0.094317079519777, 0.23833355316548, 3.3363753829239], [0.12521906040439, 0.03, 0.21516945776157,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               2.4508433350573], [0.11421019333548, 0.061499011497294, 0.41946439392279, 1.2323108869], [0.086305841200261, 0.088706042551795, 0.13813602191822, 5.1599254543833], [0.13173942942765, 0.12350256134312, 0.75, 3.9693903166633], [0.17614085214159, 0.15, 0.59507090665931, 3.2770072073404], [0.13883939826687, 0.14586950873522, 0.48208051512176, 3.6271201165173], [0.087114450179299, 0.040985602960244, 0.17124264932725, 3.0415779451434], [0.097205554354253, 0.12082764356137, 0.44722862900288, 1.9448918765068], [0.093995758725724, 0.081373801927315, 0.17820563154008, 4.5675306817912], [0.069894299327156, 0.057475808459266, 0.18097043030704, 4.4720946857941], [0.13219179363809, 0.1462469100281, 0.75, 2.4374534999852], [0.072342723460166, 0.052900744239827, 0.20380004771085, 4.4623488025485], [0.063212105996395, 0.068636263927473, 0.18594543828779, 2.7382468597687], [0.08054466586271, 0.078760972875113, 0.28711471910625, 3.7590582021091], [0.074690367660323, 0.052169725959116, 0.17030452801699, 3.0980124893767], [0.14321524616663, 0.093042848006309, 0.75, 2.9057971012588], [0.054788576973847, 0.03, 0.1, 3.1698072854378], [0.10534422720185, 0.031797235495378, 0.23495974754319, 5.0252000973009], [0.099065915948221, 0.064858514908295, 0.27187505412172, 3.0025506701765], [0.088705744000464, 0.10275058003627, 0.38073160010157, 3.0974832140969], [0.16402676447595, 0.13313608680274, 0.51319890204588, 1.166915082904], [0.13034428253682, 0.1431894187626, 0.75, 3.3550255296817], [0.10294234234813, 0.049744483601254, 0.28890534337916, 3.5696918435571], [0.12571635127086, 0.096866960205948, 0.29499479659882, 2.5411201532223], [0.1000787818189, 0.049079912062959, 0.38334020094691, 2.7126157787229], [0.1479167634974, 0.09629694280608, 0.37493989282969, 3.0165807947013], [0.070095038561427, 0.082803015683009, 0.27997680945402, 2.5095168950956], [0.071539799029628, 0.099094488419242, 0.30607331637046, 5.6755188917474], [0.088314676711907, 0.055753717297801, 0.23851770774032, 2.5351990660329], [0.10117743397404, 0.078462535737341, 0.5463657535781, 2.8197410994355], [0.16010494152036, 0.088232264042623, 0.27486269882784, 4.1664651655491], [0.059392814616617, 0.056644161744612, 0.16617306287465, 3.1515905980752], [0.13732802832017, 0.11641313529938, 0.35583122751175, 3.2965106038064], [0.098794456448179, 0.07293995056557, 0.44653160377371, 3.9966000858361], [0.13144038236467, 0.11243464170412, 0.36673401347558, 2.107709010638], [0.063882605913494, 0.039141483446014, 0.15876934389026, 2.214379480064], [0.056980628195313, 0.03, 0.1, 3.8471510758446], [0.17448335708525, 0.12795995697385, 0.75, 3.4769681354175], [0.065051974960521, 0.067859242806417, 0.22576054967965, 3.0937351722307], [0.079269329218908, 0.06688251911803, 0.17133524483365, 3.4123526221123], [0.15690107689746, 0.11365590621391, 0.41199747126945, 3.2710045559701], [0.097193631972382, 0.11938458856116, 0.32281275877487, 3.7375620092324], [0.1423527128933, 0.13287156977883, 0.75, 4.6489844501143], [0.14423993136539, 0.1418123779467, 0.40585325682191, 3.0284932210318], [0.13077543955667, 0.11436096124224, 0.38825897831714, 4.3441325277497], [0.13529414917172, 0.10079306146734, 0.41220662317937, 5.3875689912937], [0.08029077693591, 0.089949956755307, 0.23109198497505, 0.22955086284879], [0.07, 0.03, 0.1263613926807, 4.1251527025521], [0.10731236182375, 0.091328413129029, 0.20150572191532, 3.287683732231], [0.14016526264168, 0.15, 0.47820025528309, 2.4881620862859], [0.16355608355763, 0.15, 0.59541779754136, 3.5940701708032], [0.11430812759503, 0.13026803130455, 0.37454041986459, 4.0690235355497], [0.15580490224717, 0.12922673634604, 0.455438495607, 1.0766524615684], [0.1303665158626, 0.13898728203977, 0.39856732851642, 2.974130602917], [0.14079472913793, 0.11270116042956, 0.37014175878874, 5.0658012970135], [0.12170390878788, 0.15, 0.43637546134608, 4.5849465309494], [0.12512354072637, 0.098538030610358, 0.32976902323333, 4.9328468908403], [0.1363492787987, 0.15, 0.48806204193694, 3.2134639574355], [0.11850434159965, 0.11151623424643, 0.34486100219079, 4.70666048632], [0.084024664512799, 0.04972427698517, 0.12449491368036, 5.2222544100339], [0.099200676539949, 0.098970851204098, 0.28219507178034, 5.2644216179615]]