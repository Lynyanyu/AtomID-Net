import numpy as np

nms_lt = np.array([40.88115692138672, 54.88252639770508, 51.86033248901367, 53.46846580505371, 53.85422706604004, 52.397727966308594, 52.83999443054199, 54.40354347229004, 54.85224723815918, 50.83441734313965, 50.86326599121094, 52.414894104003906, 53.82680892944336, 53.663015365600586, 54.54707145690918, 52.8867244720459, 51.89037322998047, 51.86128616333008, 52.82902717590332, 52.451133728027344, 51.831960678100586, 51.86128616333008, 51.894187927246094, 51.86128616333008, 52.857398986816406, 52.38485336303711, 51.86009407043457, 50.864458084106445, 50.83441734313965, 50.408124923706055, 52.85930633544922, 50.807952880859375, 52.484750747680664, 52.85787582397461, 51.86152458190918, 52.858591079711914, 51.89228057861328, 51.862239837646484, 52.85787582397461, 51.860809326171875, 53.854942321777344, 50.862789154052734, 52.858591079711914, 52.40154266357422, 52.85954475402832, 51.862478256225586, 50.408124923706055, 49.83711242675781, 50.379276275634766, 51.86176300048828])
nms_lt2 = np.array([175.50230026245117, 170.07803916931152, 162.5652313232422, 188.0631446838379, 186.99407577514648, 183.3486557006836, 184.53478813171387, 205.53922653198242, 213.0744457244873, 177.61611938476562, 172.50967025756836, 184.7822666168213, 272.1524238586426, 237.49661445617676, 267.8418159484863, 260.1950168609619, 186.97500228881836, 211.55381202697754, 257.47013092041016, 277.4066925048828, 261.32988929748535, 287.2791290283203, 262.3739242553711, 266.2653923034668, 266.28899574279785, 290.9054756164551, 200.43611526489258, 209.19322967529297, 133.14270973205566, 180.5424690246582, 266.8769359588623, 198.01664352416992, 276.03626251220703, 275.3489017486572, 242.476224899292, 272.0012664794922, 183.53843688964844, 189.0876293182373, 269.6046829223633, 293.2155132293701, 311.89489364624023, 281.6648483276367, 304.8069477081299, 281.9387912750244, 278.5298824310303, 285.3736877441406, 205.55448532104492, 188.0340576171875, 163.36464881896973, 168.0915355682373])
nms_ch = np.array([1.9882277250289917, 2.164158344268799, 1.8772883415222168, 1.850037693977356, 1.9251790046691895, 1.7411198616027832, 1.8289005756378174, 2.031404733657837, 2.265949249267578, 4.948992729187012, 5.433237552642822, 6.0042877197265625, 2.092650890350342, 2.9835562705993652, 2.3115687370300293, 2.4024312496185303, 1.8007822036743164, 2.035175323486328, 2.3121936321258545, 2.506654977798462, 1.9516079425811768, 2.049389362335205, 2.362685441970825, 1.7460989952087402, 1.9078898429870605, 1.768533706665039, 2.2473719120025635, 2.6080453395843506, 6.571137428283691, 1.772926926612854, 1.5137076377868652, 5.357051849365234, 1.8806335926055908, 3.6481471061706543, 1.6602940559387207, 2.3088412284851074, 1.9256565570831299, 1.8977081775665283, 1.4835819005966187, 1.5144758224487305, 1.5951905250549316, 1.747981071472168, 1.5955524444580078, 1.6501582860946655, 1.6472113132476807, 1.6501582860946655, 1.6517951488494873, 1.8047091960906982, 2.471336841583252, 2.150387763977051])
nms_ja = np.array([0.9644259810447693, 0.9530751705169678, 0.9609823226928711, 0.9742765426635742, 0.9775229096412659, 0.9780370593070984, 0.9771403074264526, 0.9677708148956299, 0.9526363015174866, 0.9336514472961426, 0.9293628931045532, 0.8842471837997437, 0.9621973037719727, 0.9578260183334351, 0.9588340520858765, 0.9618449807167053, 0.9718433618545532, 0.9843189716339111, 0.9807229042053223, 0.9806013703346252, 0.9883734583854675, 0.9838441610336304, 0.9837654232978821, 0.9299273490905762, 0.9183048605918884, 0.92904132604599, 0.9258597493171692, 0.9040035605430603, 0.9912751913070679, 0.986288845539093, 0.9807395339012146, 0.9583333134651184, 0.9963477253913879, 0.9511166214942932, 0.9781460165977478, 0.980577826499939, 0.9916840195655823, 0.9976894855499268, 0.9854467511177063, 0.9843490719795227, 0.9813975691795349, 0.9797150492668152, 0.9787842631340027, 0.9769047498703003, 0.9992709755897522, 0.9769047498703003, 0.9976846575737, 0.9981498718261719, 0.9554747939109802, 0.9580387473106384])
nms_f1 = np.array([0.9864146113395691, 0.9841232895851135, 0.9861045479774475, 0.9901960492134094, 0.9873740673065186, 0.9898124933242798, 0.9905713796615601, 0.9880040884017944, 0.9826017022132874, 0.9496186375617981, 0.9386804103851318, 0.9311490654945374, 0.9551451206207275, 0.9222309589385986, 0.9491226673126221, 0.9441745281219482, 0.9918060302734375, 0.9906437397003174, 0.9514347910881042, 0.9482384324073792, 0.9647386074066162, 0.9607331156730652, 0.9509310126304626, 0.9662730097770691, 0.9591196775436401, 0.9665709137916565, 0.9788690805435181, 0.9724007248878479, 0.808318555355072, 0.9937831163406372, 0.9783549904823303, 0.7784515023231506, 0.9730115532875061, 0.9345930814743042, 0.9745767712593079, 0.9674251675605774, 0.9940950870513916, 0.9948163032531738, 0.9775752425193787, 0.9778562784194946, 0.9748297333717346, 0.9678534865379333, 0.9772118330001831, 0.973774790763855, 0.9721614718437195, 0.973774790763855, 0.995150089263916, 0.9961915612220764, 0.9780257940292358, 0.9791399836540222])

LoG_lt = np.array([548.4933853149414, 599.877119064331, 587.0966911315918, 567.2435760498047, 574.1987228393555, 581.6423892974854, 563.3845329284668, 594.7141647338867, 580.6612968444824, 613.4464740753174, 597.6386070251465, 592.975378036499, 946.9823837280273, 848.2728004455566, 851.7642021179199, 838.7851715087891, 586.0798358917236, 595.2205657958984, 850.2058982849121, 850.2693176269531, 860.2871894836426, 880.7125091552734, 860.2545261383057, 847.7051258087158, 855.8619022369385, 837.3367786407471, 578.7007808685303, 580.3172588348389, 594.963550567627, 583.7199687957764, 842.604398727417, 898.895263671875, 832.219123840332, 842.017412185669, 842.9734706878662, 866.9662475585938, 575.4964351654053, 567.7151679992676, 856.3802242279053, 843.6896800994873, 849.9782085418701, 841.9466018676758, 837.7158641815186, 864.05348777771, 857.2487831115723, 849.7555255889893, 567.3630237579346, 562.7784729003906, 586.897611618042, 583.4381580352783])
LoG_ch = np.array([2.089261531829834, 2.2206218242645264, 1.9867312908172607, 1.9677977561950684, 1.9541056156158447, 1.7965331077575684, 1.7564157247543335, 2.022697925567627, 2.2621700763702393, 4.903557777404785, 5.301670074462891, 6.010262966156006, 1.7623780965805054, 2.6735901832580566, 1.9870702028274536, 2.100302219390869, 1.7148197889328003, 1.976053237915039, 1.9496177434921265, 2.1633782386779785, 1.6744403839111328, 1.7167887687683105, 2.0045504570007324, 1.8640855550765991, 1.9357695579528809, 1.908311367034912, 2.3910365104675293, 2.7230608463287354, 4.766286373138428, 1.8634388446807861, 1.4361319541931152, 4.5923686027526855, 1.780700922012329, 3.5300986766815186, 1.541900396347046, 2.29929780960083, 2.017123222351074, 2.0319318771362305, 1.4064204692840576, 1.3853024244308472, 1.4576536417007446, 1.5768873691558838, 1.5437722206115723, 1.5354456901550293, 1.4613213539123535, 1.5354456901550293, 1.7191393375396729, 1.9416067600250244, 2.3191981315612793, 2.0635623931884766])
LoG_ja = np.array([0.9528772234916687, 0.9485675692558289, 0.9397456049919128, 0.9664322733879089, 0.9724169373512268, 0.9676331877708435, 0.9468682408332825, 0.9329608678817749, 0.9108469486236572, 0.8774744868278503, 0.8445072770118713, 0.8657633066177368, 0.9517567753791809, 0.9451383352279663, 0.9500166773796082, 0.9574177265167236, 0.9633986949920654, 0.9693520069122314, 0.9741631150245667, 0.9742110371589661, 0.982550323009491, 0.9790948629379272, 0.9790909290313721, 0.8885745406150818, 0.8749052882194519, 0.8892672657966614, 0.9230256080627441, 0.9024713039398193, 0.8749182820320129, 0.9845208525657654, 0.9765571355819702, 0.9232633113861084, 0.9953067898750305, 0.945087730884552, 0.9732657074928284, 0.975468635559082, 0.990338146686554, 0.9944916367530823, 0.9811818599700928, 0.9812399744987488, 0.9746517539024353, 0.9714774489402771, 0.9751524329185486, 0.9714774489402771, 0.9979174137115479, 0.9714774489402771, 0.9903868436813354, 0.9976931810379028, 0.933931291103363, 0.9437679648399353])
LoG_f1 = np.array([0.9869763851165771, 0.9853543639183044, 0.9831893444061279, 0.9903544783592224, 0.9905573129653931, 0.990566074848175, 0.9852792024612427, 0.9806888103485107, 0.9741203188896179, 0.9458670616149902, 0.9328573942184448, 0.9287904500961304, 0.981374204158783, 0.9480154514312744, 0.9759241938591003, 0.9695606827735901, 0.9900369644165039, 0.990604043006897, 0.9811064600944519, 0.9777500629425049, 0.9895792603492737, 0.9898819327354431, 0.9791464805603027, 0.9681664705276489, 0.9635323882102966, 0.9664517045021057, 0.9784224033355713, 0.9723047614097595, 0.928968071937561, 0.9943671822547913, 0.9920037388801575, 0.874466061592102, 0.9906575083732605, 0.948768675327301, 0.9911981821060181, 0.9872350096702576, 0.9951456189155579, 0.9957486391067505, 0.9936091303825378, 0.9932839274406433, 0.9911750555038452, 0.9895985126495361, 0.9919841289520264, 0.9906346797943115, 0.9956135749816895, 0.9906346797943115, 0.9953991174697876, 0.9971181750297546, 0.9794654250144958, 0.9824844598770142])

DoG_lt = np.array([308.93659591674805, 329.7157287597656, 325.6537914276123, 319.26918029785156, 328.33170890808105, 330.87825775146484, 337.68558502197266, 317.58999824523926, 338.1938934326172, 333.27627182006836, 328.4494876861572, 328.9957046508789, 645.1280117034912, 590.6085968017578, 609.1532707214355, 603.1713485717773, 334.4094753265381, 336.31110191345215, 618.0598735809326, 603.4572124481201, 696.0403919219971, 697.9198455810547, 614.0878200531006, 610.7659339904785, 609.5857620239258, 612.89381980896, 320.934534072876, 329.82325553894043, 309.2031478881836, 329.7579288482666, 629.0922164916992, 442.6701068878174, 613.8241291046143, 466.37773513793945, 628.7391185760498, 596.9293117523193, 317.6867961883545, 330.8436870574951, 610.9559535980225, 605.8261394500732, 630.9208869934082, 606.8294048309326, 612.2982501983643, 615.8566474914551, 632.469892501831, 613.987922668457, 326.8260955810547, 323.6651420593262, 330.4438591003418, 331.2373161315918])
DoG_ch = np.array([1.9764931201934814, 2.114671230316162, 1.8518000841140747, 1.9004900455474854, 1.8901898860931396, 1.7015457153320312, 1.7223725318908691, 1.9880332946777344, 2.2211132049560547, 5.0008745193481445, 5.5204668045043945, 6.199216365814209, 1.827301025390625, 3.075226306915283, 2.110659122467041, 2.373300075531006, 1.6671756505966187, 1.9551196098327637, 2.15627384185791, 2.448888063430786, 1.7287487983703613, 1.7893528938293457, 2.315439224243164, 1.8648656606674194, 1.9580447673797607, 1.9309629201889038, 2.336486339569092, 2.6862478256225586, 5.4811811447143555, 1.787514090538025, 1.4524829387664795, 5.331207275390625, 1.9192728996276855, 4.1475934982299805, 1.5690724849700928, 2.385082721710205, 1.9546265602111816, 1.969954490661621, 1.4258332252502441, 1.409754991531372, 1.4868645668029785, 1.6261911392211914, 1.5483148097991943, 1.5500112771987915, 1.508333444595337, 1.5500112771987915, 1.6345176696777344, 1.8853740692138672, 2.1488521099090576, 1.9166128635406494])
DoG_ja = np.array([0.9646139740943909, 0.9588571190834045, 0.9567493796348572, 0.9711736440658569, 0.9781171679496765, 0.9728568196296692, 0.9642384052276611, 0.9498680830001831, 0.9383260011672974, 0.9113980531692505, 0.8967683911323547, 0.8779301047325134, 0.9647085666656494, 0.9702944159507751, 0.9661017060279846, 0.9744157195091248, 0.9709698557853699, 0.9812750816345215, 0.9875805974006653, 0.9865919351577759, 0.9890298247337341, 0.984718918800354, 0.9889827370643616, 0.901012122631073, 0.8883824348449707, 0.902614951133728, 0.9253964424133301, 0.9066044092178345, 0.9434500932693481, 0.9881033897399902, 0.9787455201148987, 0.9543317556381226, 0.9985454678535461, 0.9690420031547546, 0.9756125807762146, 0.9828264713287354, 0.9912320971488953, 0.9972324967384338, 0.9824642539024353, 0.9827665686607361, 0.9768029451370239, 0.9752772450447083, 0.9760009050369263, 0.9733578562736511, 0.9988323450088501, 0.9733578562736511, 0.9967697262763977, 0.9976900219917297, 0.958502471446991, 0.9628202319145203])
DoG_f1 = np.array([0.9891611337661743, 0.9869426488876343, 0.9868513941764832, 0.9897411465644836, 0.9908786416053772, 0.9914707541465759, 0.9881235361099243, 0.9828233122825623, 0.9782983064651489, 0.9420156478881836, 0.926608681678772, 0.9184499382972717, 0.9751999378204346, 0.9088334441184998, 0.9632867574691772, 0.9444444179534912, 0.991243839263916, 0.990772008895874, 0.96090167760849, 0.949999988079071, 0.9838581085205078, 0.9816404581069946, 0.9496326446533203, 0.9680348634719849, 0.9620755910873413, 0.9652262926101685, 0.9783917665481567, 0.9719871878623962, 0.8993570804595947, 0.9945883750915527, 0.9911101460456848, 0.7915331721305847, 0.9768186211585999, 0.8907831907272339, 0.9891675710678101, 0.9778700470924377, 0.9943293333053589, 0.9953958988189697, 0.9919049739837646, 0.9922857880592346, 0.9899907112121582, 0.9863301515579224, 0.9911535978317261, 0.9888584613800049, 0.9917106628417969, 0.9888584613800049, 0.9961950182914734, 0.9964240193367004, 0.9839101433753967, 0.9857259392738342])

DoH_lt = np.array([942.3151016235352, 958.2631587982178, 924.6425628662109, 993.0145740509033, 1000.1955032348633, 1010.265588760376, 1044.8179244995117, 1056.9195747375488, 1015.765905380249, 1168.7989234924316, 1094.8905944824219, 1145.296335220337, 1622.1551895141602, 1035.9084606170654, 1291.4659976959229, 1128.4384727478027, 1112.5569343566895, 1058.1090450286865, 1156.45432472229, 1148.0305194854736, 1497.1189498901367, 1399.5330333709717, 1126.891851425171, 1482.5448989868164, 1476.7930507659912, 1495.1059818267822, 981.351375579834, 1025.2654552459717, 860.1300716400146, 1013.0434036254883, 1451.7171382904053, 1020.6167697906494, 1231.1384677886963, 826.7719745635986, 1481.1756610870361, 1401.2269973754883, 1000.1986026763916, 1027.5335311889648, 1447.6206302642822, 1433.8712692260742, 1480.818748474121, 1469.3512916564941, 1531.8589210510254, 1520.1451778411865, 1468.2352542877197, 1488.1260395050049, 954.4050693511963, 1035.282850265503, 1008.1627368927002, 994.7450160980225])
DoH_ch = np.array([3.3442676067352295, 3.4848227500915527, 3.3425240516662598, 3.4477500915527344, 3.485616683959961, 3.2266645431518555, 3.491772174835205, 3.6227760314941406, 3.724788188934326, 6.584015369415283, 6.923623085021973, 7.237425804138184, 3.6914687156677246, 5.5107011795043945, 4.071369171142578, 4.808836460113525, 3.3807549476623535, 3.619257926940918, 4.602749347686768, 4.8495707511901855, 3.8341143131256104, 3.935009002685547, 4.873342514038086, 3.4896578788757324, 3.7812509536743164, 3.7273383140563965, 3.646671772003174, 3.965282440185547, 8.074249267578125, 3.224207878112793, 3.0862348079681396, 7.907382965087891, 4.2338151931762695, 7.468178749084473, 3.152597665786743, 4.037966728210449, 3.229736804962158, 3.2529706954956055, 3.090233087539673, 3.09626841545105, 3.216118574142456, 3.375749111175537, 3.0109200477600098, 3.1442298889160156, 3.2030529975891113, 3.1442298889160156, 3.1955413818359375, 3.2161664962768555, 3.4452569484710693, 3.231156349182129])
DoH_ja = np.array([0.9788334965705872, 0.9737950563430786, 0.9738745093345642, 0.9796107411384583, 0.9842774271965027, 0.9794097542762756, 0.9778431057929993, 0.9716830253601074, 0.9682539701461792, 0.9357106685638428, 0.9424650073051453, 0.8909178376197815, 0.9809876680374146, 0.9834938049316406, 0.9793758988380432, 0.9853249192237854, 0.9773333072662354, 0.9867700934410095, 0.9894002676010132, 0.9861805438995361, 0.9911417365074158, 0.9835622906684875, 0.9899635314941406, 0.9151175022125244, 0.9106104373931885, 0.9142342209815979, 0.926008939743042, 0.9087640643119812, 0.9890859723091125, 0.9894471168518066, 0.9802026152610779, 0.9586535692214966, 0.9991552829742432, 0.9693945646286011, 0.97724848985672, 0.9859472513198853, 0.9916685819625854, 0.9967570304870605, 0.9830488562583923, 0.9831116199493408, 0.9780885577201843, 0.9779561758041382, 0.9774392247200012, 0.9743739366531372, 0.9995190501213074, 0.9743739366531372, 0.9962980151176453, 0.9967592358589172, 0.9776588082313538, 0.9804012775421143])
DoH_f1 = np.array([0.9885985255241394, 0.9869575500488281, 0.9880487322807312, 0.9899320602416992, 0.9885057210922241, 0.9908574819564819, 0.9867669939994812, 0.9810929298400879, 0.9731378555297852, 0.9029444456100464, 0.884020209312439, 0.8898488879203796, 0.9474750757217407, 0.7879873514175415, 0.9061958193778992, 0.8557122945785522, 0.9909869432449341, 0.983517050743103, 0.8854771256446838, 0.8574715852737427, 0.948713481426239, 0.9446276426315308, 0.8435920476913452, 0.9556193351745605, 0.9393977522850037, 0.9442341327667236, 0.9770522713661194, 0.9687387943267822, 0.799118161201477, 0.9942370653152466, 0.9882007837295532, 0.5911579728126526, 0.9027539491653442, 0.6733939051628113, 0.9804434180259705, 0.9277694821357727, 0.9931625127792358, 0.992847204208374, 0.9891932606697083, 0.9891759753227234, 0.9836488366127014, 0.9703994393348694, 0.9883583784103394, 0.9826148748397827, 0.9775974750518799, 0.9826148748397827, 0.9944572448730469, 0.994687020778656, 0.9831499457359314, 0.9849976301193237])

print("nms_cuda_lt: ", nms_lt.mean(), " +- ", nms_lt.std())
print("nms_cpu_lt: ", nms_lt2.mean(), " +- ", nms_lt2.std())
print("nms_ch: ", nms_ch.mean(), " +- ", nms_ch.std())
print("nms_ja: ", nms_ja.mean(), " +- ", nms_ja.std())
print("nms_f1: ", nms_f1.mean(), " +- ", nms_f1.std())

print("LoG_lt: ", LoG_lt.mean(), " +- ", LoG_lt.std())
print("LoG_ch: ", LoG_ch.min(), " +- ", LoG_ch.std())
print("LoG_ja: ", LoG_ja.mean(), " +- ", LoG_ja.std())
print("LoG_f1: ", LoG_f1.mean(), " +- ", LoG_f1.std())

print("DoG_lt: ", DoG_lt.mean(), " +- ", DoG_lt.std())
print("DoG_ch: ", DoG_ch.min(), " +- ", DoG_ch.std())
print("DoG_ja: ", DoG_ja.mean(), " +- ", DoG_ja.std())
print("DoG_f1: ", DoG_f1.mean(), " +- ", DoG_f1.std())

print("DoH_lt: ", DoH_lt.mean(), " +- ", DoH_lt.std())
print("DoH_ch: ", DoH_ch.min(), " +- ", DoH_ch.std())
print("DoH_ja: ", DoH_ja.mean(), " +- ", DoH_ja.std())
print("DoH_f1: ", DoH_f1.mean(), " +- ", DoH_f1.std())