from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas
from urllib.parse import quote

OUT = "/mnt/data/Lane_Detection_Stitching_Paper_Research.pdf"

# Use broadly available fonts without shipping font files.
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='Title2', parent=styles['Title'], fontSize=24, leading=28, alignment=TA_CENTER, textColor=HexColor('#17365D'), spaceAfter=10))
styles.add(ParagraphStyle(name='SubTitle', parent=styles['Normal'], fontSize=11, leading=15, alignment=TA_CENTER, textColor=HexColor('#555555'), spaceAfter=16))
styles.add(ParagraphStyle(name='H1x', parent=styles['Heading1'], fontSize=17, leading=21, textColor=HexColor('#17365D'), spaceBefore=8, spaceAfter=8))
styles.add(ParagraphStyle(name='H2x', parent=styles['Heading2'], fontSize=13, leading=17, textColor=HexColor('#245A8D'), spaceBefore=6, spaceAfter=5))
styles.add(ParagraphStyle(name='Bodyx', parent=styles['BodyText'], fontSize=9.2, leading=13, spaceAfter=6))
styles.add(ParagraphStyle(name='Smallx', parent=styles['BodyText'], fontSize=7.7, leading=10, textColor=HexColor('#555555'), spaceAfter=4))
styles.add(ParagraphStyle(name='Callout', parent=styles['BodyText'], fontSize=9.3, leading=13, backColor=HexColor('#EEF4FA'), borderColor=HexColor('#B8CBE0'), borderWidth=0.5, borderPadding=7, spaceBefore=5, spaceAfter=8))
styles.add(ParagraphStyle(name='PaperTitle', parent=styles['Heading2'], fontSize=13.5, leading=17, textColor=HexColor('#17365D'), spaceBefore=2, spaceAfter=4))
styles.add(ParagraphStyle(name='Label', parent=styles['BodyText'], fontSize=8.3, leading=11, textColor=HexColor('#666666')))
styles.add(ParagraphStyle(name='TableCell', parent=styles['BodyText'], fontSize=7.2, leading=9.1))
styles.add(ParagraphStyle(name='TableHead', parent=styles['BodyText'], fontSize=7.3, leading=9.2, textColor=colors.white, alignment=TA_LEFT))

papers = [
    {
        'title':'Ultra Fast Structure-aware Deep Lane Detection (UFLD)', 'year':'2020', 'venue':'ECCV 2020',
        'link':'https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/4584_ECCV_2020_paper.php',
        'code':'https://github.com/cfzd/Ultra-Fast-Lane-Detection',
        'about':'Recasts lane detection as row-wise location selection over global features instead of dense pixel segmentation. Adds a structural loss to encourage lane shape consistency.',
        'uses':'Front-view RGB; row-based lane representation; global context; structural loss. Evaluated on CULane and TuSimple.',
        'adv':'Very fast and lightweight; designed for difficult occlusion/lighting while keeping computation low. The paper reports a lightweight version above 300 FPS on its stated hardware.',
        'draw':'The representation is tied to row-wise sampling and is strongest for conventional forward-view lanes. It does not solve persistent lane identity, topology, or Indian-domain adaptation by itself.',
        'novel':'Speed-oriented row-selection formulation plus explicit structural modeling of lane shape.',
        'improve':'Use as a speed baseline or lightweight front-end, then add robust polyline fitting, BEV geometry, temporal stitching, and confidence calibration. For your project, do not rely on UFLD alone for merge/split topology.',
        'fit':'HIGH as a lightweight baseline; MEDIUM as final detector.'
    },
    {
        'title':'End-to-end Lane Shape Prediction with Transformers (LSTR)', 'year':'2021', 'venue':'WACV 2021',
        'link':'https://arxiv.org/abs/2011.04233', 'code':'https://github.com/liuruijin17/LSTR',
        'about':'Uses a transformer to directly predict lane-shape parameters rather than relying on a feature extractor followed by heavy post-processing.',
        'uses':'Transformer encoder/decoder-style reasoning; compact lane shape parameters with physical interpretation tied to road structure/camera pose; TuSimple emphasis.',
        'adv':'Extremely small model and low compute footprint. The project reports 765,787 parameters and very low MACs, making it interesting for edge deployment.',
        'draw':'Primarily a single-frame lane-shape predictor; complex topology, persistent lane IDs, and general multi-dataset training are outside its core scope.',
        'novel':'Direct lane-shape parameter prediction with transformer context in a very small model.',
        'improve':'Add sequence memory and lane-ID state, plus confidence based on geometric and temporal consistency. Use as an efficiency benchmark on the M4.',
        'fit':'HIGH for latency experiments; MEDIUM for final research stack.'
    },
    {
        'title':'CondLaneNet: A Top-To-Down Lane Detection Framework Based on Conditional Convolution', 'year':'2021', 'venue':'ICCV 2021',
        'link':'https://openaccess.thecvf.com/content/ICCV2021/html/Liu_CondLaneNet_A_Top-To-Down_Lane_Detection_Framework_Based_on_Conditional_Convolution_ICCV_2021_paper.html', 'code':'https://github.com/aliyun/conditional-lane-detection',
        'about':'Detects lane instances first, then dynamically predicts their shapes using conditional convolution. Adds a Recurrent Instance Module for dense and forked lane structures.',
        'uses':'Conditional convolution, row-wise lane formulation, recurrent instance modeling. Benchmarked on three lane datasets.',
        'adv':'Designed explicitly for complex topologies and reports a strong accuracy/speed combination; the paper reports 78.14 F1 and 220 FPS on CULane under its protocol.',
        'draw':'Still mainly frame-based; persistent lane identity and explicit BEV temporal stitching are not the main target.',
        'novel':'Top-to-down instance-first detection plus dynamic lane-shape prediction and recurrence for complex topology.',
        'improve':'Feed temporal context into the instance representation, then stitch lane instances using BEV centerline/tangent/curvature similarity. Use topology constraints for Y/merge/split cases.',
        'fit':'HIGH candidate for complex-lane baseline.'
    },
    {
        'title':'Structured Bird\'s-Eye-View Traffic Scene Understanding from Onboard Images (STSU)', 'year':'2021', 'venue':'ICCV 2021',
        'link':'https://www.research-collection.ethz.ch/handle/20.500.11850/554616', 'code':'https://github.com/ybarancan/STSU',
        'about':'Builds a directed road-network graph directly in BEV from a single forward-facing camera image and can also represent dynamic objects on the ground plane.',
        'uses':'Monocular onboard camera; BEV road graph; lane curves and intersections; object semantics/orientation. Supports nuScenes and Argoverse in released code.',
        'adv':'Moves beyond isolated lane lines toward structured road topology and object-road interaction.',
        'draw':'Heavy compared with simple lane detectors; depends on BEV/scene-graph formulation and is not tailored to Indian road appearance or lightweight M4 deployment.',
        'novel':'Explicit BEV directed road graph from onboard images.',
        'improve':'Use the graph idea without importing the full model: maintain persistent lane IDs and a lightweight lane graph over time; fuse with your tracked objects for lane relation.',
        'fit':'HIGH for architecture/novelty inspiration, not necessarily the production detector.'
    },
    {
        'title':'LaneFormer: Object-Aware Row-Column Transformers for Lane Detection', 'year':'2022', 'venue':'AAAI 2022',
        'link':'https://ojs.aaai.org/index.php/AAAI/article/view/19961', 'code':'',
        'about':'Transformer lane detector that uses row/column attention to capture lane shape context and explicitly incorporates detected object instances into the network.',
        'uses':'Deformable pixel-wise attention; row/column self-attention; object bounding boxes and ROI-aligned features; CULane and TuSimple experiments.',
        'adv':'Directly models the fact that vehicles/pedestrians can occlude or contextualize lane markings. Reported 77.1% F1 on CULane.',
        'draw':'More complex than lightweight classical or row-selection detectors; integrating a separate YOLO/tracker stream increases engineering complexity.',
        'novel':'Object-aware lane detection - objects become part of lane prediction context.',
        'improve':'This is highly relevant to your stack: use your existing tracked-object boxes as temporal occlusion context, but keep object interaction as an optional refinement stage rather than replacing the detector.',
        'fit':'HIGH for your fusion research angle.'
    },
    {
        'title':'CLRNet: Cross Layer Refinement Network for Lane Detection', 'year':'2022', 'venue':'CVPR 2022',
        'link':'https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_CLRNet_Cross_Layer_Refinement_Network_for_Lane_Detection_CVPR_2022_paper.html', 'code':'https://github.com/Turoad/CLRNet',
        'about':'Uses high-level semantic features for initial lane proposals and low-level features for localization refinement. Introduces ROIGather and whole-lane Line IoU loss.',
        'uses':'Anchor-based lane representation; cross-layer refinement; ROIGather; Line IoU loss. Official implementation supports CULane, TuSimple and LLAMAS.',
        'adv':'Strong frame-level accuracy and a useful whole-lane regression objective; provides a strong starting point for modern anchor-based lane detection.',
        'draw':'Confidence calibration is not its central contribution; frame-level detection still leaves temporal stitching and topology to downstream systems.',
        'novel':'Cross-layer refinement with high-level context plus low-level localization, and Line IoU for whole-lane structure.',
        'improve':'Add a temporal consistency head or post-hoc lane tracker, use Indian-domain adaptation, and calibrate confidence across time rather than treating frame scores as final certainty.',
        'fit':'HIGH as production detector baseline.'
    },
    {
        'title':'A Keypoint-Based Global Association Network for Lane Detection (GANet)', 'year':'2022', 'venue':'CVPR 2022',
        'link':'https://openaccess.thecvf.com/content/CVPR2022/html/Wang_A_Keypoint-Based_Global_Association_Network_for_Lane_Detection_CVPR_2022_paper.html', 'code':'https://github.com/Wolfwjs/GANet',
        'about':'Represents lanes as keypoints and directly associates keypoints with lane starts, avoiding inefficient point-by-point grouping.',
        'uses':'Keypoint estimation plus global association. Released CULane and TuSimple models using ResNet18/34/101 backbones.',
        'adv':'Flexible lane-shape representation and strong speed/accuracy tradeoffs. Official repo reports 153 FPS for the ResNet18 CULane model on its stated setup.',
        'draw':'The global association is within a frame; persistent lane identity across time is not the primary problem formulation.',
        'novel':'Global keypoint-to-lane association instead of sequential grouping.',
        'improve':'Reuse the global association concept temporally: associate lane observations across frames using keypoint geometry, not only fitted curves. This may reduce ID fragmentation in curved lanes.',
        'fit':'HIGH alternative detector baseline.'
    },
    {
        'title':'PersFormer: 3D Lane Detection via Perspective Transformer and the OpenLane Benchmark', 'year':'2022', 'venue':'ECCV 2022',
        'link':'https://arxiv.org/abs/2203.11089', 'code':'https://github.com/OpenDriveLab/PersFormer_3DLane',
        'about':'End-to-end monocular 3D lane detection using a transformer-based spatial feature transformation from perspective to BEV, with unified 2D/3D anchors.',
        'uses':'Camera parameters; transformer spatial feature transformation; 2D/3D lane supervision; OpenLane and Apollo 3D Lane Synthetic.',
        'adv':'Directly tackles 3D lane geometry and camera-view transformation. OpenLane provides a large real-world 3D lane benchmark.',
        'draw':'More demanding than a simple 2D lane detector and depends on camera/geometry assumptions; deployment on your Mac requires careful benchmarking.',
        'novel':'Perspective Transformer for 3D lanes plus the OpenLane benchmark.',
        'improve':'Use its BEV geometry idea but avoid heavy end-to-end 3D complexity initially: infer 2D lanes, transform into a practical BEV, then perform temporal stitching there.',
        'fit':'HIGH for BEV research; MEDIUM for first production detector.'
    },
    {
        'title':'Topology Preserving Local Road Network Estimation From Single Onboard Camera Image', 'year':'2022', 'venue':'CVPR 2022',
        'link':'https://openaccess.thecvf.com/content/CVPR2022/html/Can_Topology_Preserving_Local_Road_Network_Estimation_From_Single_Onboard_Camera_CVPR_2022_paper.html', 'code':'https://github.com/ybarancan/TopologicalLaneGraph',
        'about':'Represents the road network as directed lane curves and their interactions in BEV. Introduces minimal cycles and covers to represent topology.',
        'uses':'Single onboard camera; directed lane curves; intersection points; minimal-cycle topology representation; NuScenes and Argoverse evaluation.',
        'adv':'Explicitly models the graph structure needed for merges, splits and intersections instead of treating every lane independently.',
        'draw':'Single-image topology reasoning is not the same as temporal lane ID tracking; it is also more complex than the minimal real-time stack you currently need.',
        'novel':'Minimal cycles/covers as a structured representation of local road topology from one image.',
        'improve':'Use a simplified temporal graph: persistent lane nodes + directed edges + merge/split events, with lane IDs carried frame-to-frame.',
        'fit':'VERY HIGH for stitching/topology design.'
    },
    {
        'title':'Learning to Predict 3D Lane Shape and Camera Pose from a Single Image via Geometry Constraints', 'year':'2022', 'venue':'AAAI 2022',
        'link':'https://ojs.aaai.org/index.php/AAAI/article/view/20069', 'code':'https://github.com/liuruijin17/CLGo',
        'about':'Predicts camera pose from a single image and uses geometry constraints to create a top-view for more accurate 3D lane prediction without ground-truth camera pose.',
        'uses':'Two-stage pose + 3D lane estimation; geometry constraints; transformer context; polynomial 3D lane representation.',
        'adv':'Important because your repository currently has no camera calibration. It shows that learned camera pose can substitute for perfect calibration in some settings.',
        'draw':'Adds another learned estimation problem; errors in camera pose can contaminate BEV geometry.',
        'novel':'Joint camera pose and lane geometry consistency for monocular 3D lane estimation.',
        'improve':'For your project, first try a calibrated or fixed projective transform if defensible; only move to learned pose estimation if calibration uncertainty demonstrably harms stitching.',
        'fit':'HIGH if calibration becomes a bottleneck.'
    },
    {
        'title':'BEV-LaneDet: An Efficient 3D Lane Detection Based on Virtual Camera via Key-Points', 'year':'2023', 'venue':'CVPR 2023',
        'link':'https://openaccess.thecvf.com/content/CVPR2023/html/Wang_BEV-LaneDet_An_Efficient_3D_Lane_Detection_Based_on_Virtual_Camera_CVPR_2023_paper.html', 'code':'https://github.com/gigo-team/bev_lane_det',
        'about':'Efficient monocular 3D lane detector using a virtual camera, key-point representation and a lightweight spatial transformation pyramid into BEV features.',
        'uses':'Virtual camera; key-point 3D lane representation; spatial transformation pyramid; OpenLane and Apollo 3D Lane Synthetic.',
        'adv':'Explicitly optimized for practical efficiency; paper reports strong 3D lane performance and 185 FPS on its stated hardware.',
        'draw':'3D/BEV model complexity may be unnecessary until your 2D detector and temporal stitching are stable. Hardware speed claims are not transferable directly to an M4.',
        'novel':'Virtual-camera normalization + key-point 3D lanes + efficient BEV transformation.',
        'improve':'Adopt the virtual-camera normalization concept for cross-dataset consistency, then perform lane stitching in BEV with lightweight temporal association.',
        'fit':'VERY HIGH for BEV direction; benchmark before adoption.'
    },
    {
        'title':'OpenLane-V2: A Topology Reasoning Benchmark for Unified 3D HD Mapping', 'year':'2023', 'venue':'NeurIPS 2023 Datasets and Benchmarks',
        'link':'https://proceedings.neurips.cc/paper_files/paper/2023/hash/3c0a4c8c236144f1b99b7e1531debe9c-Abstract.html', 'code':'https://github.com/OpenDriveLab/OpenLane-V2',
        'about':'Benchmark for scene topology that connects 3D lanes with traffic elements and lane-lane relationships, moving beyond isolated lane detection.',
        'uses':'OpenLane-derived 3D lane task plus traffic-element/lane topology. Includes lane-lane and lane-traffic relationships.',
        'adv':"Directly relevant to your mentor's merge/split/intersection and object-to-lane association goals.",
        'draw':'Only 2,000 annotated road scenes in the topology benchmark; topology reasoning is substantially more demanding than simple lane-line detection.',
        'novel':'A unified perception-and-reasoning benchmark for lane/traffic topology.',
        'improve':'Use its representation ideas, but make your system lightweight: persistent lane IDs, directed relations, and object-to-lane edges at runtime.',
        'fit':'VERY HIGH for research framing and topology.'
    },
    {
        'title':'TopoNet: Graph-based Topology Reasoning for Driving Scenes', 'year':'2023', 'venue':'2023 preprint; later journal publication in 2026',
        'link':'https://arxiv.org/abs/2304.05277', 'code':'https://github.com/OpenDriveLab/TopoNet',
        'about':'Builds a scene knowledge graph to reason about lane-lane connectivity and lane-traffic-element assignment beyond conventional perception.',
        'uses':'Semantic embedding, scene graph neural network and knowledge-graph style relation modeling on OpenLane-V2.',
        'adv':'Shows how topology can be treated as a first-class reasoning layer rather than a side-effect of detection.',
        'draw':'Graph reasoning is heavier than needed for a simple lane tracker and depends on strong upstream detection.',
        'novel':'Explicit scene-graph reasoning for driving topology.',
        'improve':'Use a simpler deterministic graph over your persistent lane IDs: nodes are lane segments and edges encode continuation/merge/split; then attach tracked objects to lanes.',
        'fit':'HIGH for novelty and fusion design.'
    },
    {
        'title':'LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving', 'year':'2024', 'venue':'ICLR 2024',
        'link':'https://openreview.net/pdf?id=LsURkIPYR5', 'code':'https://github.com/OpenDriveLab/LaneSegNet',
        'about':'Introduces lane segments as a representation that jointly carries geometry and topology, aiming for a complete road-structure map.',
        'uses':'Lane attention; reference-point initialization; OpenLane-V2 tasks including map element detection, centerline perception and lane segment perception.',
        'adv':'Strongly aligned with merge/split/topology requirements while providing a single object (lane segment) that carries geometry and connectivity.',
        'draw':'The released training recipe recommends multi-GPU training and the reported 14.7 FPS is on its stated environment, so it may be too close to your total 12 FPS floor when combined with other models.',
        'novel':'Lane segment as a joint geometry + topology representation.',
        'improve':'Use lane segments as the conceptual output schema, but decouple expensive detection from cheap temporal stitching. A lightweight detector can emit segments that your stitcher links over time.',
        'fit':'VERY HIGH for research direction; benchmark carefully for real-time.'
    },
    {
        'title':'CLRerNet: Improving Confidence of Lane Detection With LaneIoU', 'year':'2024', 'venue':'WACV 2024',
        'link':'https://openaccess.thecvf.com/content/WACV2024/html/Honda_CLRerNet_Improving_Confidence_of_Lane_Detection_With_LaneIoU_WACV_2024_paper.html', 'code':'https://github.com/hirotomusiker/CLRerNet',
        'about':'Improves the usefulness of lane confidence by introducing LaneIoU, which accounts for local lane angles and is used in target assignment and loss functions.',
        'uses':'Anchor-based CLRNet family; LaneIoU target assignment/loss; CULane and CurveLanes evaluation.',
        'adv':"Directly addresses your mentor's confidence requirement. Reports 81.43% F1 on CULane and 86.47% on CurveLanes under its benchmark setup.",
        'draw':'Confidence is improved for frame-level detection, but it does not itself guarantee temporal confidence calibration or persistent lane identity.',
        'novel':'LaneIoU used specifically to improve alignment between lane confidence and actual lane quality.',
        'improve':'Use CLRerNet confidence as the raw detector score, then calibrate final lane confidence with temporal persistence, BEV geometry agreement, visibility and stitching consistency. Never make confidence rise merely because an object is physically closer.',
        'fit':'TOP CHOICE for your learned 2D detector baseline.'
    },
    {
        'title':'Lane Graph as Path: Continuity-Preserving Path-Wise Modeling for Online Lane Graph Construction (LaneGAP)', 'year':'2024', 'venue':'ECCV 2024',
        'link':'https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/6068_ECCV_2024_paper.php', 'code':'https://github.com/hustvl/LaneGAP',
        'about':'Models the lane graph as continuous paths rather than small pieces, arguing that path continuity matches how driving actually proceeds. Uses Path2Graph recovery.',
        'uses':'Path-wise lane representation; end-to-end path prediction; Path2Graph graph recovery; nuScenes, Argoverse2, OpenLane-V2 comparisons.',
        'adv':'Directly attacks the continuity problem that your stitching module is trying to solve.',
        'draw':'Focuses on online lane graph construction rather than your simpler 2D lane-marking detector plus persistent IDs; may be heavier than needed.',
        'novel':'Path as the primitive unit of lane-graph construction, preserving continuity and traffic-flow semantics.',
        'improve':'Borrow the path-continuity principle: stitch lanes using long-range path evidence, not only adjacent-frame overlap. This is probably the strongest conceptual ingredient for your novelty.',
        'fit':'VERY HIGH for your stitching research.'
    },
]



general_papers = [
    {
        'title':'Three Decades of Driver Assistance Systems: Review and Future Perspectives', 'year':'2014', 'venue':'IEEE Intelligent Transportation Systems Magazine',
        'link':'https://doi.org/10.1109/MITS.2014.2336271',
        'about':'Historical review of driver assistance systems, their goals, development path and future perspectives. It is useful for positioning the project in the progression from warning systems toward automated driving.',
        'uses':'System-level ADAS taxonomy, driver assistance functions, sensing, human-machine interaction and development perspectives.',
        'adv':'Excellent background for explaining why perception must be coupled to driver assistance functions rather than treated as an isolated vision problem.',
        'draw':'A review rather than a new perception algorithm; predates current deep-learning, BEV and large-scale multimodal benchmarks.',
        'novel':'Its value is the historical/system framing across decades of ADAS evolution rather than a single model.',
        'improve':'Use it as the historical anchor, then show how your modular camera-first perception stack modernizes the sensing and fusion layers for Indian-road conditions.',
        'fit':'VERY HIGH for project motivation and literature introduction.'
    },
    {
        'title':'Embedding Vision-based Advanced Driver Assistance Systems: A Survey', 'year':'2017', 'venue':'IET Intelligent Transport Systems',
        'link':'https://doi.org/10.1049/iet-its.2016.0026',
        'about':'Survey focused on embedding vision-based ADAS in vehicles, bridging algorithms with hardware, software, testing and practical deployment constraints.',
        'uses':'Camera-based ADAS functions, embedded implementation considerations, hardware/software choices, safety and testing requirements.',
        'adv':'Directly relevant to your embedded/realtime requirement and helps justify why latency, robustness and verification are first-class constraints.',
        'draw':'Published before the current deep-perception/BEV wave; does not cover modern lane transformers, monocular depth transformers or current multimodal fusion.',
        'novel':'Bridges computer-vision algorithms with practical embedded-ADAS engineering.',
        'improve':'Use its deployment framing and extend it with modern asynchronous perception, confidence propagation, temporal state and Indian-domain validation.',
        'fit':'TOP-TIER for your ADAS architecture rationale.'
    },
    {
        'title':'Vision-based Driver Assistance Systems: Survey, Taxonomy and Advances', 'year':'2021', 'venue':'arXiv survey / later research literature',
        'link':'https://arxiv.org/abs/2104.12583',
        'about':'Broad survey and taxonomy of vision-based driver assistance systems, covering computer vision, machine learning, embedded systems, automotive electronics and safety-critical software.',
        'uses':'System taxonomy, ADAS functions and the relationship between perception, embedded implementation and deployment.',
        'adv':'Useful bridge between classical ADAS and modern learning-based perception; gives a vocabulary for positioning your project.',
        'draw':'Survey rather than a single benchmarked architecture; not targeted to the exact Indian-road task.',
        'novel':'Provides a consistent taxonomy and top-down abstraction for scaling vision ADAS toward autonomy.',
        'improve':'Use its taxonomy to map your detector, tracker, depth, segmentation, lane and risk modules into one traceable safety/perception pipeline.',
        'fit':'VERY HIGH for system-level framing.'
    },
    {
        'title':'A New Multi-Camera Approach for Lane Departure Warning', 'year':'2011', 'venue':'Advanced Concepts for Intelligent Vision Systems',
        'link':'https://people-ece.vse.gmu.edu/~hayes/papers/ACIVS_2011.pdf',
        'about':'Camera-based lane departure warning system using multiple cameras, perspective removal, calibration, lane detection and fusion to estimate vehicle-to-lane-boundary distance.',
        'uses':'Multi-camera imaging, bird\'s-eye-view transformation, camera calibration, lane detection and distance-to-boundary estimation.',
        'adv':'A strong pre-deep-learning reference showing that BEV geometry and calibration were already central to practical LDW.',
        'draw':'Multi-camera rather than your single forward RGB stream; classical detection and no learned temporal lane identity.',
        'novel':'Combines multiple camera views in BEV and fuses lane estimates to perform LDW.',
        'improve':'Adapt the BEV/calibration concept to a single camera, then add learned lane detection and persistent temporal lane IDs with confidence.',
        'fit':'VERY HIGH for historical BEV/LDW lineage.'
    },
    {
        'title':'Computer Vision for Autonomous Vehicles: Problems, Datasets and State of the Art', 'year':'2017', 'venue':'Survey / arXiv',
        'link':'https://arxiv.org/abs/1704.05519',
        'about':'Broad survey of computer vision for autonomous vehicles covering recognition, reconstruction, motion estimation, tracking, scene understanding, datasets and end-to-end driving.',
        'uses':'KITTI, MOT, Cityscapes and other benchmarks; taxonomy across perception tasks and open problems.',
        'adv':'Excellent literature map and historical baseline for explaining why the project decomposes perception into multiple tasks.',
        'draw':'Pre-BEVFormer/modern transformer era and therefore not sufficient alone for current lane/BEV literature.',
        'novel':'Comprehensive problem-and-dataset survey for AV vision rather than a single model.',
        'improve':'Use it as the broad foundation, then update its missing modern layers with BEV, temporal transformers, topology, monocular depth and large-scale datasets.',
        'fit':'TOP-TIER background reference.'
    },
    {
        'title':'DeepDriving: Learning Affordance for Direct Perception in Autonomous Driving', 'year':'2015', 'venue':'ICCV 2015',
        'link':'https://doi.org/10.1109/ICCV.2015.312',
        'about':'Introduces a direct-perception formulation that predicts compact driving affordances from images instead of mapping directly from image to steering.',
        'uses':'Deep CNN; compact scene affordances; synthetic driving data; KITTI-based car distance estimation.',
        'adv':'Important conceptual precursor to your risk/affordance layer: perception can output compact variables directly related to driving decisions.',
        'draw':'Early deep-learning approach; limited compared with modern dense/temporal perception and does not solve robust multi-object scene understanding.',
        'novel':'A middle ground between full mediated perception and end-to-end control via explicit driving affordances.',
        'improve':'Use the affordance idea for your downstream risk representation: TTC, lane conflict, closing speed and ego-lane relation become compact actionable state.',
        'fit':'HIGH for risk/fusion design.'
    },
    {
        'title':'Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite', 'year':'2012', 'venue':'CVPR 2012',
        'link':'https://doi.org/10.1109/cvpr.2012.6248074',
        'about':'Foundational autonomous-driving benchmark introducing challenging real-world tasks across stereo, optical flow, visual odometry/SLAM and 3D object detection.',
        'uses':'Stereo cameras, Velodyne lidar and localization; real driving sequences and 3D annotations.',
        'adv':'Established the principle that perception must be evaluated under real driving conditions, not laboratory imagery.',
        'draw':'Older sensor suite and benchmark scope; largely structured compared with Indian road environments and modern multimodal tasks.',
        'novel':'A demanding, real-world benchmark suite that helped standardize AV perception evaluation.',
        'improve':'Use the same benchmark philosophy for your project: evaluate on hard Indian scenes and report failure modes, not only average accuracy.',
        'fit':'VERY HIGH as historical perception benchmark context.'
    },
    {
        'title':'nuScenes: A Multimodal Dataset for Autonomous Driving', 'year':'2020', 'venue':'CVPR 2020',
        'link':'https://openaccess.thecvf.com/content_CVPR_2020/html/Caesar_nuScenes_A_Multimodal_Dataset_for_Autonomous_Driving_CVPR_2020_paper.html',
        'about':'Large-scale multimodal AV dataset covering cameras, lidar, radar and 3D annotations, with tasks for detection and tracking.',
        'uses':'6 cameras, 5 radars, 1 lidar, GPS/IMU context, 3D boxes, tracking and scene-level attributes.',
        'adv':'Demonstrates why aligned multimodal streams and temporal context matter for robust AV perception.',
        'draw':'Our project is deliberately camera-first and does not have lidar/radar ground truth; full nuScenes-style models are too heavy for the M4 + 12 FPS constraint.',
        'novel':'First dataset with the full AV sensor suite at large scale plus 3D detection/tracking benchmarks.',
        'improve':'Borrow the synchronized, timestamped multi-stream design but replace expensive sensors with monocular depth and asynchronous learned modules.',
        'fit':'HIGH for fusion architecture and benchmark context.'
    },
    {
        'title':'BEVFormer: Learning Bird\'s-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers', 'year':'2022', 'venue':'ECCV 2022',
        'link':'https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/694_ECCV_2022_paper.php',
        'about':'Camera-only BEV framework that uses spatial cross-attention and temporal self-attention to build a persistent bird\'s-eye-view representation.',
        'uses':'Multi-camera images, BEV queries, spatial cross-attention and temporal history.',
        'adv':'Direct evidence that temporal BEV state can improve perception and velocity estimation under low visibility.',
        'draw':'Designed for multi-camera AV rigs and computationally heavier than a lightweight single-camera lane module.',
        'novel':'Unified spatial-temporal BEV representation learned from camera images.',
        'improve':'Use a lightweight analogue: transform lane polylines into BEV and maintain a temporal lane state rather than a full BEV transformer.',
        'fit':'VERY HIGH for BEV + temporal design inspiration.'
    },
    {
        'title':'BEVFusion: A Simple and Robust LiDAR-Camera Fusion Framework', 'year':'2022', 'venue':'NeurIPS 2022',
        'link':'https://arxiv.org/abs/2205.13790',
        'about':'A robust sensor-fusion architecture that unifies camera and lidar features in BEV and studies robustness to missing lidar.',
        'uses':'Camera and lidar features, BEV shared representation and robustness training.',
        'adv':'Strong argument for a common BEV representation as the natural meeting point of different perception streams.',
        'draw':'Requires lidar in the original task and is far beyond what is needed for your monocular prototype.',
        'novel':'Camera stream can remain useful even under lidar malfunction through BEV-level fusion.',
        'improve':'Adopt the architectural principle only: make your camera-derived lane/depth/object signals meet in one common scene-coordinate representation before risk fusion.',
        'fit':'HIGH for fusion architecture, not a model to deploy directly.'
    },
    {
        'title':'Depth Anything V2', 'year':'2024', 'venue':'arXiv / computer vision literature',
        'link':'https://arxiv.org/abs/2406.09414',
        'about':'Large-scale monocular depth model family emphasizing fine, robust relative depth through synthetic supervision, larger teachers and pseudo-labelled real images.',
        'uses':'RGB input; teacher-student training; synthetic data; pseudo-labelled real images; multiple model sizes.',
        'adv':'Provides practical monocular depth without lidar and has multiple sizes for speed/accuracy trade-offs.',
        'draw':'Depth is relative unless metric scale is recovered; our MPS benchmark shows the small model is an auxiliary low-rate stream rather than a 30 FPS stage.',
        'novel':'Large-scale synthetic/pseudo-labelled training strategy and scalable monocular depth family.',
        'improve':'Use V2 Small asynchronously and combine depth trend with tracking, trajectory and lane conflict instead of treating one depth frame as ground truth.',
        'fit':'TOP-TIER for your existing depth module.'
    },
    {
        'title':'ByteTrack: Multi-Object Tracking by Associating Every Detection Box', 'year':'2022', 'venue':'ECCV 2022',
        'link':'https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/315_ECCV_2022_paper.php',
        'about':'Tracking-by-detection method that keeps low-confidence detections in a second association stage to recover occluded objects and reduce fragmentation.',
        'uses':'Detector boxes, high/low confidence association, tracklets and IoU-style matching.',
        'adv':'Simple and fast; directly targets fragmented tracks caused by discarding low-confidence detections.',
        'draw':'Our actual video has severe same-class duplicate detections and ownership swaps, so generic ByteTrack alone was insufficient in testing.',
        'novel':'Associating almost every detection instead of only high-confidence boxes.',
        'improve':'Use ByteTrack as a conceptual baseline, but add duplicate suppression, class-aware ownership, temporal inertia and task-specific identity persistence.',
        'fit':'HIGH as tracking literature context.'
    },
    {
        'title':'Deep SORT: Simple Online and Realtime Tracking with a Deep Association Metric', 'year':'2017', 'venue':'ICIP 2017',
        'link':'https://arxiv.org/abs/1703.07402',
        'about':'Extends SORT with a learned visual appearance metric to preserve identities through longer occlusions.',
        'uses':'Kalman filtering, nearest-neighbor appearance embeddings and track-to-detection association.',
        'adv':'Classic demonstration that appearance can reduce identity switches during occlusion.',
        'draw':'Appearance inference adds compute and can be unreliable when different road users look similar; our project-level ReID experiment provided little benefit per FPS cost.',
        'novel':'Deep appearance metric integrated into an otherwise simple online tracker.',
        'improve':'Use appearance only as a gated fallback for genuinely ambiguous cases, with geometry/trajectory remaining primary.',
        'fit':'HIGH for tracking theory and comparison.'
    },
]

# Dataset/source summary
sources = [
    ('CULane','https://xingangpan.github.io/projects/CULane.html','Difficult lane scenarios, occlusion, shadow, dazzle, curve, night; strong 2D benchmark.'),
    ('CurveLanes','https://github.com/SoulmateB/CurveLanes','Curves, complex geometry, multi-lane/curved cases.'),
    ('BDD100K','https://github.com/bdd100k/bdd100k','Weather/time diversity; lane marking categories; solid/dashed attributes.'),
    ('OpenLane','https://github.com/OpenDriveLab/OpenLane','Large real-world 3D lane benchmark and temporal lane structure.'),
    ('OpenLane-V2','https://github.com/OpenDriveLab/OpenLane-V2','Lane topology and lane-traffic relationships.'),
    ('IDD','https://idd.insaan.iiit.ac.in/','Indian road domain adaptation/evaluation.'),
    ('GTSDB','https://benchmark.ini.rub.de/','German traffic-sign detection.'),
    ('GTSRB','https://benchmark.ini.rub.de/','German traffic-sign recognition.'),
    ('highD','https://www.highd-dataset.com/','German highway trajectories/lane changes; not image lane masks.'),
    ('inD','https://www.ind-dataset.com/','German intersection trajectories; not direct camera lane labels.'),
]

scenario = [
    ('Straight','CULane, BDD100K, TuSimple','Strong'),
    ('Curved / S-curve','CurveLanes, CULane','Strong'),
    ('Dashed / Solid','BDD100K','Strong'),
    ('Double white / yellow','BDD100K','Useful'),
    ('Faded / worn','BDD100K/CULane selected hard cases','Verify sample coverage'),
    ('Partially missing / occluded','CULane','Strong'),
    ('Shadow / dazzle / night','CULane','Strong'),
    ('Merge / split / Y-junction','OpenLane-V2, CurveLanes, selected BDD100K','Best for topology'),
    ('Intersection / crossing','OpenLane-V2, inD','Topology/trajectory'),
    ('Arrow-marked road','BDD100K','Useful'),
    ('Multiple lanes','CurveLanes, OpenLane','Strong'),
    ('Wet / poor visibility','BDD100K + targeted evaluation','Verify exact labels'),
    ('Crowded traffic','CULane, BDD100K','Strong'),
]

story = []
story.append(Spacer(1, 12))
story.append(Paragraph('Lane Detection, Stitching and Automotive Perception - Paper Research', styles['Title2']))
story.append(Paragraph('Research synthesis for an Indian-road automotive perception pipeline with a hard >=12 FPS requirement', styles['SubTitle']))
story.append(Paragraph('<b>Purpose.</b> This document compares influential and directly relevant lane-perception papers, identifies what each contributes, what it does not solve, and turns the literature into a practical architecture for your project. “Drawbacks” and “How we can improve” are engineering assessments based on the stated scope of each work; they are not claims made verbatim by the authors.', styles['Callout']))
story.append(Paragraph('Recommended project direction', styles['H1x']))
story.append(Paragraph('<b>Best first learned detector benchmark:</b> CLRerNet. It is the closest match to your mentor\'s confidence requirement and has published results on both CULane and CurveLanes. <b>Best BEV/geometry references:</b> PersFormer, BEV-LaneDet, and geometry-constrained camera-pose work. <b>Best topology references:</b> STSU, Topology Preserving Local Road Network Estimation, OpenLane-V2, LaneSegNet, TopoNet and LaneGAP. <b>Best efficiency baselines:</b> UFLD and LSTR. <b>Best object-aware connection to your existing stack:</b> LaneFormer.', styles['Bodyx']))
story.append(Paragraph('Suggested end-to-end research pipeline', styles['H1x']))
pipe = 'CULane + CurveLanes + BDD100K + OpenLane/OpenLane-V2 + IDD<br/>-&gt; canonical lane representation<br/>-&gt; learned lane detector (CLRerNet first benchmark)<br/>-&gt; Indian-domain adaptation<br/>-&gt; optional BEV normalization<br/>-&gt; temporal path/geometry stitching<br/>-&gt; persistent lane IDs<br/>-&gt; ego-lane + lane topology<br/>-&gt; object-to-lane association<br/>-&gt; fusion / risk'
story.append(Paragraph(pipe, styles['Callout']))
story.append(Paragraph('Research questions to answer', styles['H1x']))
questions = [
    'Can a learned detector reduce the false lane hypotheses of the current HSV/Canny/Hough baseline without violating the real-time budget?',
    'Does BEV normalization improve temporal lane matching enough to justify its compute and calibration complexity?',
    'Can persistent lane IDs be made more stable by matching continuous lane paths rather than frame-local lane segments?',
    'Can final lane confidence be calibrated from detector confidence + temporal continuity + BEV/geometric consistency instead of using raw detector confidence?',
    'Can the existing YOLO/tracker object stream be used as an occlusion/context signal for lane detection, inspired by LaneFormer?',
]
for q in questions:
    story.append(Paragraph('&bull; ' + q, styles['Bodyx']))
story.append(PageBreak())

story.append(Paragraph('Paper-by-paper review', styles['H1x']))
for i,p in enumerate(papers, start=1):
    story.append(Paragraph(f'{i}. {p["title"]}', styles['PaperTitle']))
    meta = f'<b>Year:</b> {p["year"]} &nbsp;&nbsp; <b>Venue:</b> {p["venue"]} &nbsp;&nbsp; <b>Fit:</b> {p["fit"]}'
    story.append(Paragraph(meta, styles['Bodyx']))
    links = f'<b>Paper:</b> <link href="{p["link"]}" color="#1A73E8">{p["link"]}</link>'
    if p['code']:
        links += f' &nbsp;&nbsp; <b>Code:</b> <link href="{p["code"]}" color="#1A73E8">{p["code"]}</link>'
    story.append(Paragraph(links, styles['Smallx']))
    for label,key in [('What it is about','about'),('What it uses','uses'),('Advantages','adv'),('Drawbacks / boundary','draw'),('Novelty','novel'),('How we can improve / reuse it','improve')]:
        story.append(Paragraph(f'<b>{label}:</b> {p[key]}', styles['Bodyx']))
    if i in [5,9,14,15]:
        story.append(Spacer(1, 4))
    if i in [4,8,12]:
        story.append(PageBreak())

story.append(PageBreak())
story.append(Paragraph('Cross-paper comparison', styles['H1x']))
headers = ['Paper / line of work','Primary representation','Temporal ID?','BEV / 3D?','Topology?','Confidence focus?','Real-time relevance']
rows = [[Paragraph(h, styles['TableHead']) for h in headers]]
for p in papers:
    title = p['title'].split(':')[0][:42]
    rep = ('Row/select' if 'Ultra Fast' in p['title'] else
           'Shape params' if 'LSTR' in p['title'] else
           'Conditional lane instance' if 'CondLaneNet' in p['title'] else
           'BEV graph' if 'Structured Bird' in p['title'] else
           'Object-aware transformer' if 'LaneFormer' in p['title'] else
           'Anchor/Line-IoU' if 'CLRNet' in p['title'] else
           'Keypoint/global assoc.' if 'GANet' in p['title'] else
           '2D/3D anchors' if 'PersFormer' in p['title'] else
           'Directed lane graph' if 'Topology Preserving' in p['title'] else
           '3D polynomial lanes' if 'Learning to Predict 3D' in p['title'] else
           '3D keypoints / BEV' if 'BEV-LaneDet' in p['title'] else
           'Lane/traffic topology' if 'OpenLane-V2' in p['title'] else
           'Scene graph' if 'TopoNet' in p['title'] else
           'Lane segments' if 'LaneSegNet' in p['title'] else
           'Path-wise graph')
    temp = 'No' if p['title'] not in ['Lane Graph as Path: Continuity-Preserving Path-Wise Modeling for Online Lane Graph Construction (LaneGAP)'] else 'Online path'
    bev = 'Yes' if any(s in p['title'] for s in ['Structured Bird','PersFormer','Topology Preserving','Learning to Predict 3D','BEV-LaneDet','OpenLane-V2','LaneSegNet','TopoNet','Lane Graph']) else 'No / indirect'
    topo = 'Yes' if any(s in p['title'] for s in ['Structured Bird','Topology Preserving','OpenLane-V2','TopoNet','LaneSegNet','Lane Graph']) else 'Limited'
    conf = 'Core' if 'CLRerNet' in p['title'] else 'Secondary'
    realtime = p['fit'].split(' for')[0]
    rows.append([Paragraph(str(x), styles['TableCell']) for x in [title,rep,temp,bev,topo,conf,realtime]])

t = Table(rows, colWidths=[38*mm,31*mm,19*mm,22*mm,20*mm,21*mm,28*mm], repeatRows=1)
t.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),HexColor('#245A8D')),('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),0.25,HexColor('#CBD5E1')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,HexColor('#F8FAFC')]),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)
]))
story.append(t)
story.append(Spacer(1, 10))
story.append(Paragraph('Important conclusion: the literature largely treats <i>detection</i>, <i>BEV/3D geometry</i>, <i>topology</i>, and <i>online lane-graph construction</i> as separate or partially connected problems. That leaves room for a lightweight engineering pipeline that explicitly carries persistent lane IDs and confidence across time while consuming a learned lane detector.', styles['Callout']))

story.append(PageBreak())
story.append(Paragraph('General Automotive Perception + Pre-ADAS Camera Literature', styles['H1x']))
story.append(Paragraph('These references give the project a second literature axis beyond lane detection: the evolution of camera-based ADAS, embedded automotive perception, scene understanding, temporal tracking, BEV representations, multimodal fusion and driving affordances. The improvement notes below are engineering assessments for your system, not claims that the original authors made.', styles['Bodyx']))
for i,p in enumerate(general_papers, start=1):
    story.append(Paragraph(f'{i}. {p["title"]}', styles['PaperTitle']))
    story.append(Paragraph(f'<b>Year:</b> {p["year"]} &nbsp;&nbsp; <b>Venue:</b> {p["venue"]} &nbsp;&nbsp; <b>Fit:</b> {p["fit"]}', styles['Bodyx']))
    story.append(Paragraph(f'<b>Paper:</b> <link href="{p["link"]}" color="#1A73E8">{p["link"]}</link>', styles['Smallx']))
    for label,key in [('What it is about','about'),('What it uses','uses'),('Advantages','adv'),('Drawbacks / boundary','draw'),('Novelty','novel'),('How we can improve / reuse it','improve')]:
        story.append(Paragraph(f'<b>{label}:</b> {p[key]}', styles['Bodyx']))
    if i in [4,8,12]:
        story.append(PageBreak())

story.append(PageBreak())
story.append(Paragraph('How the literature maps to your current system', styles['H1x']))
rows = [[Paragraph(x, styles['TableHead']) for x in ['Existing project block','Key literature','What it contributes','What your project adds']]]
map_rows = [
    ('Object detection','YOLO (2016) + modern real-time detectors','Real-time object localization and class confidence.','Persistent identity, duplicate suppression and domain-specific ownership behavior on the Indian dashcam.'),
    ('Object tracking','Deep SORT (2017), ByteTrack (2022)','Temporal association, occlusion recovery, low-confidence box usage.','Lightweight ownership hysteresis, duplicate-track suppression and explicit long/short lifecycle logic.'),
    ('Depth','Depth Anything V2 (2024)','Strong monocular relative depth from RGB with scalable model sizes.','Asynchronous cached depth attached to persistent tracks and used for closing-rate/TTC reasoning.'),
    ('Lane detection','CLRNet / CLRerNet / CondLaneNet / GANet','Modern lane localization, confidence and complex-shape modeling.','Indian-domain adaptation plus persistent lane IDs and downstream lane-object fusion.'),
    ('BEV / geometry','STSU, PersFormer, BEV-LaneDet, BEVFormer','Scene-coordinate normalization, 3D/BEV geometry and temporal representation.','Use BEV as a lightweight geometric state for lane stitching rather than a heavyweight full-scene transformer.'),
    ('Topology','OpenLane-V2, TopoNet, LaneSegNet, LaneGAP, UniTopo (2026)','Lane connectivity, lane segments, graph/path representations and topology reasoning.','Persistent lane identity + topology + ego-lane relation in a lightweight real-time pipeline.'),
    ('Risk / affordance','DeepDriving + ADAS surveys','Compact actionable scene variables and system-level ADAS framing.','TTC + closing speed + trajectory conflict + lane relation under asynchronous camera-first perception.'),
    ('System engineering','Bengler 2014; Velez 2017; Horgan 2021','Embedded constraints, safety, testing and the evolution of vision ADAS.','Explicit >=12 FPS gate, asynchronous workers, confidence/staleness handling and Indian-road validation.'),
]
for r in map_rows:
    rows.append([Paragraph(str(x), styles['TableCell']) for x in r])
t=Table(rows,colWidths=[30*mm,48*mm,57*mm,58*mm],repeatRows=1)
t.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),HexColor('#245A8D')),('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),0.25,HexColor('#CBD5E1')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,HexColor('#F8FAFC')]),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)
]))
story.append(t)
story.append(Spacer(1,8))
story.append(Paragraph('<b>Current frontier note:</b> UniTopo (2026) is a useful latest reference because it moves toward directly modeling lane topology rather than treating topology as only a downstream relation between already detected lanes. Your proposed lane stitching remains a different engineering objective: persistent lane identity and temporal continuity under a lightweight camera-first pipeline.', styles['Callout']))
story.append(Paragraph('<b>Paper:</b> <link href="https://ieeexplore.ieee.org/document/11506407" color="#1A73E8">https://ieeexplore.ieee.org/document/11506407</link>', styles['Smallx']))

story.append(Paragraph('Dataset and scenario strategy', styles['H1x']))
story.append(Paragraph('Do not merge every dataset into one training pool blindly. Normalize each source into a canonical lane representation first, and assign each dataset a role.', styles['Bodyx']))
rows = [[Paragraph(x, styles['TableHead']) for x in ['Dataset','Role','Why it matters']]]
for name,url,desc in sources:
    rows.append([Paragraph(f'<b>{name}</b>', styles['TableCell']), Paragraph(f'<link href="{url}" color="#1A73E8">{url}</link>', styles['TableCell']), Paragraph(desc, styles['TableCell'])])
t = Table(rows, colWidths=[30*mm,63*mm,86*mm], repeatRows=1)
t.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),HexColor('#245A8D')),('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),0.25,HexColor('#CBD5E1')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,HexColor('#F8FAFC')]),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)
]))
story.append(t)
story.append(PageBreak())

story.append(Paragraph('Scenario coverage matrix', styles['H1x']))
rows = [[Paragraph(x, styles['TableHead']) for x in ['Scenario','Best source(s)','Assessment']]]
for row in scenario:
    rows.append([Paragraph(str(x), styles['TableCell']) for x in row])
t = Table(rows, colWidths=[47*mm,75*mm,57*mm], repeatRows=1)
t.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),HexColor('#245A8D')),('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),0.25,HexColor('#CBD5E1')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,HexColor('#F8FAFC')]),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)
]))
story.append(t)

story.append(Paragraph('Recommended project novelty', styles['H1x']))
story.append(Paragraph('<b>Do not claim a brand-new lane detector.</b> A defensible novelty statement is the integration and temporal formulation:', styles['Bodyx']))
story.append(Paragraph('<b>Learned lane detection -> Indian-domain adaptation -> BEV-normalized temporal path stitching -> persistent lane IDs -> topology-aware merge/split handling -> confidence calibrated by detection + temporal + geometric consistency -> ego-lane and tracked-object association.</b>', styles['Callout']))
for s in [
    '<b>Novelty axis 1 - temporal lane identity:</b> treat the lane as a persistent entity rather than a new set of curves every frame.',
    '<b>Novelty axis 2 - path-wise continuity:</b> use multiple frames, lane shape, tangent, curvature and BEV geometry to retain continuity through partial occlusion and noisy detections.',
    '<b>Novelty axis 3 - confidence calibration:</b> final lane confidence should reflect detector score, visibility, temporal persistence and geometric/BEV agreement. Proximity should affect downstream risk, not fabricate detector confidence.',
    '<b>Novelty axis 4 - object-lane coupling:</b> consume your existing YOLO/tracker tracks to reason about occluded lane markings and to assign tracked objects to ego/adjacent lanes.',
    '<b>Novelty axis 5 - Indian adaptation:</b> evaluate specifically on Indian road appearance and less-structured traffic, instead of assuming European/US benchmarks transfer directly.',
]:
    story.append(Paragraph('&bull; ' + s, styles['Bodyx']))

story.append(Paragraph('Recommended experimental ladder', styles['H1x']))
steps = [
    '1. Current HSV/Canny/Hough lane system = fixed baseline.',
    '2. Benchmark one learned 2D detector - CLRerNet first; keep GANet/UFLD as alternative speed baselines.',
    '3. Adapt/fine-tune on Indian data after canonical label normalization.',
    '4. Add BEV/IPM only after measuring whether it improves centerline/stitching stability.',
    '5. Build temporal lane stitching as a separate module using path continuity, geometry and confidence history.',
    '6. Add lane topology state for merge/split/Y-junction/intersection cases.',
    '7. Add ego-lane and object-to-lane relation for fusion.',
    '8. Evaluate the complete asynchronous pipeline at >=12 FPS, preferably >=15 FPS.',
]
for s in steps:
    story.append(Paragraph(s, styles['Bodyx']))

story.append(PageBreak())
story.append(Paragraph('Key paper links and sources', styles['H1x']))
for p in papers:
    story.append(Paragraph(f'<b>{p["title"]}</b> ({p["year"]}, {p["venue"]}) - <link href="{p["link"]}" color="#1A73E8">paper</link>' + (f' - <link href="{p["code"]}" color="#1A73E8">code</link>' if p['code'] else ''), styles['Smallx']))

story.append(Spacer(1, 10))
story.append(Paragraph('General automotive perception / ADAS links', styles['H1x']))
story.append(Paragraph('<b>UniTopo (2026)</b> - <link href="https://ieeexplore.ieee.org/document/11506407" color="#1A73E8">paper</link>', styles['Smallx']))
for p in general_papers:
    story.append(Paragraph(f'<b>{p["title"]}</b> ({p["year"]}) - <link href="{p["link"]}" color="#1A73E8">paper</link>', styles['Smallx']))
story.append(Spacer(1, 6))
story.append(Paragraph('Source notes', styles['H1x']))
story.append(Paragraph('This research brief was assembled from official conference/journal pages, official project repositories, and dataset pages. Performance numbers are kept tied to the hardware/protocol stated by the original sources; they should not be treated as expected M4 performance. Publication year is the conference/journal year unless a work is explicitly described as a preprint.', styles['Smallx']))

# footer
class FooterCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []
    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()
    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)
    def draw_page_number(self, page_count):
        self.setFont('Helvetica', 7)
        self.setFillColor(HexColor('#666666'))
        self.drawString(18*mm, 10*mm, 'Lane + automotive perception research brief')
        self.drawRightString(192*mm, 10*mm, f'Page {self._pageNumber} of {page_count}')


doc = SimpleDocTemplate(OUT, pagesize=A4, rightMargin=15*mm, leftMargin=15*mm, topMargin=15*mm, bottomMargin=16*mm, title='Lane Detection, Stitching and Automotive Perception Research')
doc.build(story, canvasmaker=FooterCanvas)
print(OUT)
