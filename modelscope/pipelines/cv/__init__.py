# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .action_detection_pipeline import ActionDetectionPipeline
    from .action_recognition_pipeline import ActionRecognitionPipeline
    from .animal_recognition_pipeline import AnimalRecognitionPipeline
    from .anydoor_pipeline import AnydoorPipeline
    from .arc_face_recognition_pipeline import ArcFaceRecognitionPipeline
    from .bad_image_detecting_pipeline import BadImageDetecingPipeline
    from .body_2d_keypoints_pipeline import Body2DKeypointsPipeline
    from .body_3d_keypoints_pipeline import Body3DKeypointsPipeline
    from .card_detection_correction_pipeline import \
        CardDetectionCorrectionPipeline
    from .card_detection_pipeline import CardDetectionPipeline
    from .cmdssl_video_embedding_pipeline import CMDSSLVideoEmbeddingPipeline
    from .controllable_image_generation_pipeline import \
        ControllableImageGenerationPipeline
    from .crowd_counting_pipeline import CrowdCountingPipeline
    from .ddcolor_image_colorization_pipeline import \
        DDColorImageColorizationPipeline
    from .ddpm_semantic_segmentation_pipeline import \
        DDPMImageSemanticSegmentationPipeline
    from .face_attribute_recognition_pipeline import \
        FaceAttributeRecognitionPipeline
    from .face_detection_pipeline import FaceDetectionPipeline
    from .face_image_generation_pipeline import FaceImageGenerationPipeline
    from .face_liveness_ir_pipeline import FaceLivenessIrPipeline
    from .face_processing_base_pipeline import FaceProcessingBasePipeline
    from .face_recognition_onnx_fm_pipeline import \
        FaceRecognitionOnnxFmPipeline
    from .face_recognition_onnx_ir_pipeline import \
        FaceRecognitionOnnxIrPipeline
    from .face_recognition_ood_pipeline import FaceRecognitionOodPipeline
    from .face_recognition_pipeline import FaceRecognitionPipeline
    from .face_reconstruction_pipeline import FaceReconstructionPipeline
    from .facial_expression_recognition_pipeline import \
        FacialExpressionRecognitionPipeline
    from .facial_landmark_confidence_pipeline import \
        FacialLandmarkConfidencePipeline
    from .general_recognition_pipeline import GeneralRecognitionPipeline
    from .hand_static_pipeline import HandStaticPipeline
    from .hicossl_video_embedding_pipeline import HICOSSLVideoEmbeddingPipeline
    from .human3d_animation_pipeline import Human3DAnimationPipeline
    from .human3d_render_pipeline import Human3DRenderPipeline
    from .human_normal_estimation_pipeline import HumanNormalEstimationPipeline
    from .image_bts_depth_estimation_pipeline import \
        ImageBTSDepthEstimationPipeline
    from .image_cartoon_pipeline import ImageCartoonPipeline
    from .image_classification_pipeline import \
        GeneralImageClassificationPipeline
    from .image_color_enhance_pipeline import ImageColorEnhancePipeline
    from .image_colorization_pipeline import ImageColorizationPipeline
    from .image_deblur_pipeline import ImageDeblurPipeline
    from .image_defrcn_fewshot_pipeline import ImageDefrcnDetectionPipeline
    from .image_denoise_pipeline import ImageDenoisePipeline
    from .image_depth_estimation_marigold_pipeline import \
        ImageDepthEstimationMarigoldPipeline
    from .image_detection_pipeline import ImageDetectionPipeline
    from .image_driving_perception_pipeline import \
        ImageDrivingPerceptionPipeline
    from .image_editing_pipeline import ImageEditingPipeline
    from .image_human_parsing_pipeline import ImageHumanParsingPipeline
    from .image_inpainting_pipeline import ImageInpaintingPipeline
    from .image_inpainting_sdv2_pipeline import ImageInpaintingSDV2Pipeline
    from .image_instance_segmentation_pipeline import \
        ImageInstanceSegmentationPipeline
    from .image_local_feature_matching_pipeline import \
        ImageLocalFeatureMatchingPipeline
    from .image_matching_fast_pipeline import ImageMatchingFastPipeline
    from .image_matching_pipeline import ImageMatchingPipeline
    from .image_matting_pipeline import ImageMattingPipeline
    from .image_mvs_depth_estimation_pipeline import \
        ImageMultiViewDepthEstimationPipeline
    from .image_open_vocabulary_detection_pipeline import \
        ImageOpenVocabularyDetectionPipeline
    from .image_paintbyexample_pipeline import ImagePaintbyexamplePipeline
    from .image_panoptic_segmentation_pipeline import \
        ImagePanopticSegmentationPipeline
    from .image_portrait_enhancement_pipeline import \
        ImagePortraitEnhancementPipeline
    from .image_quality_assessment_degradation_pipeline import \
        ImageQualityAssessmentDegradationPipeline
    from .image_quality_assessment_man_pipeline import \
        ImageQualityAssessmentMANPipeline
    from .image_quality_assessment_mos_pipeline import \
        ImageQualityAssessmentMosPipeline
    from .image_reid_person_pipeline import ImageReidPersonPipeline
    from .image_salient_detection_pipeline import ImageSalientDetectionPipeline
    from .image_semantic_segmentation_pipeline import \
        ImageSemanticSegmentationPipeline
    from .image_skychange_pipeline import ImageSkychangePipeline
    from .image_structured_model_probing_pipeline import \
        ImageStructuredModelProbingPipeline
    from .image_style_transfer_pipeline import ImageStyleTransferPipeline
    from .image_super_resolution_pasd_pipeline import \
        ImageSuperResolutionPASDPipeline
    from .image_super_resolution_pipeline import ImageSuperResolutionPipeline
    from .image_to_3d_pipeline import Image23DPipeline
    from .image_to_image_generate_pipeline import Image2ImageGenerationPipeline
    from .image_to_image_translation_pipeline import \
        Image2ImageTranslationPipeline
    from .language_guided_video_summarization_pipeline import \
        LanguageGuidedVideoSummarizationPipeline
    from .license_plate_detection_pipeline import LicensePlateDetectionPipeline
    from .lineless_table_recognition_pipeline import \
        LinelessTableRecognitionPipeline
    from .live_category_pipeline import LiveCategoryPipeline
    from .mask_face_recognition_pipeline import MaskFaceRecognitionPipeline
    from .maskdino_instance_segmentation_pipeline import \
        MaskDINOInstanceSegmentationPipeline
    from .mobile_image_super_resolution_pipeline import \
        MobileImageSuperResolutionPipeline
    from .mog_face_detection_pipeline import MogFaceDetectionPipeline
    from .movie_scene_segmentation_pipeline import \
        MovieSceneSegmentationPipeline
    from .mtcnn_face_detection_pipeline import MtcnnFaceDetectionPipelin
    from .nerf_recon_4k_pipeline import NeRFRecon4KPipeline
    from .nerf_recon_acc_pipeline import NeRFReconAccPipeline
    from .object_detection_3d_pipeline import ObjectDetection3DPipeline
    from .ocr_detection_pipeline import OCRDetectionPipeline
    from .ocr_recognition_pipeline import OCRRecognitionPipeline
    from .panorama_depth_estimation_pipeline import \
        PanoramaDepthEstimationPipeline
    from .pedestrian_attribute_recognition_pipeline import \
        PedestrainAttributeRecognitionPipeline
    from .pointcloud_sceneflow_estimation_pipeline import \
        PointCloudSceneFlowEstimationPipeline
    from .product_retrieval_embedding_pipeline import \
        ProductRetrievalEmbeddingPipeline
    from .referring_video_object_segmentation_pipeline import \
        ReferringVideoObjectSegmentationPipeline
    from .retina_face_detection_pipeline import RetinaFaceDetectionPipeline
    from .rife_video_frame_interpolation_pipeline import \
        RIFEVideoFrameInterpolationPipeline
    from .self_supervised_depth_completion_pipeline import \
        SelfSupervisedDepthCompletionPipeline
    from .shop_segmentation_pipleline import ShopSegmentationPipeline
    from .skin_retouching_pipeline import SkinRetouchingPipeline
    from .surface_recon_common_pipeline import SurfaceReconCommonPipeline
    from .table_recognition_pipeline import TableRecognitionPipeline
    from .text_driven_segmentation_pipleline import \
        TextDrivenSegmentationPipeline
    from .text_to_360panorama_image_pipeline import \
        Text2360PanoramaImagePipeline
    from .tinynas_classification_pipeline import TinynasClassificationPipeline
    from .ulfd_face_detection_pipeline import UlfdFaceDetectionPipeline
    from .video_category_pipeline import VideoCategoryPipeline
    from .video_colorization_pipeline import VideoColorizationPipeline
    from .video_deinterlace_pipeline import VideoDeinterlacePipeline
    from .video_frame_interpolation_pipeline import \
        VideoFrameInterpolationPipeline
    from .video_object_segmentation_pipeline import \
        VideoObjectSegmentationPipeline
    from .video_stabilization_pipeline import VideoStabilizationPipeline
    from .video_super_resolution_pipeline import VideoSuperResolutionPipeline
    from .vidt_pipeline import VidtPipeline
    from .virtual_try_on_pipeline import VirtualTryonPipeline
    from .vision_efficient_tuning_adapter_pipeline import \
        VisionEfficientTuningAdapterPipeline
    from .vision_efficient_tuning_lora_pipeline import \
        VisionEfficientTuningLoRAPipeline
    from .vision_efficient_tuning_prefix_pipeline import \
        VisionEfficientTuningPrefixPipeline
    from .vision_efficient_tuning_prompt_pipeline import \
        VisionEfficientTuningPromptPipeline
    from .vision_middleware_pipeline import VisionMiddlewarePipeline
    from .vop_retrieval_pipeline import VopRetrievalPipeline
    from .vop_retrieval_se_pipeline import VopRetrievalSEPipeline

else:
    _import_structure = {
        'action_recognition_pipeline': ['ActionRecognitionPipeline'],
        'action_detection_pipeline': ['ActionDetectionPipeline'],
        'animal_recognition_pipeline': ['AnimalRecognitionPipeline'],
        'body_2d_keypoints_pipeline': ['Body2DKeypointsPipeline'],
        'body_3d_keypoints_pipeline': ['Body3DKeypointsPipeline'],
        'card_detection_pipeline': ['CardDetectionPipeline'],
        'cmdssl_video_embedding_pipeline': ['CMDSSLVideoEmbeddingPipeline'],
        'hicossl_video_embedding_pipeline': ['HICOSSLVideoEmbeddingPipeline'],
        'crowd_counting_pipeline': ['CrowdCountingPipeline'],
        'image_detection_pipeline': ['ImageDetectionPipeline'],
        'image_salient_detection_pipeline': ['ImageSalientDetectionPipeline'],
        'face_detection_pipeline': ['FaceDetectionPipeline'],
        'face_image_generation_pipeline': ['FaceImageGenerationPipeline'],
        'face_recognition_pipeline': ['FaceRecognitionPipeline'],
        'face_recognition_ood_pipeline': ['FaceRecognitionOodPipeline'],
        'arc_face_recognition_pipeline': ['ArcFaceRecognitionPipeline'],
        'mask_face_recognition_pipeline': ['MaskFaceRecognitionPipeline'],
        'face_recognition_onnx_ir_pipeline': ['FaceRecognitionOnnxIrPipeline'],
        'face_recognition_onnx_fm_pipeline': ['FaceRecognitionOnnxFmPipeline'],
        'general_recognition_pipeline': ['GeneralRecognitionPipeline'],
        'image_classification_pipeline':
        ['GeneralImageClassificationPipeline'],
        'image_cartoon_pipeline': ['ImageCartoonPipeline'],
        'image_denoise_pipeline': ['ImageDenoisePipeline'],
        'image_deblur_pipeline': ['ImageDeblurPipeline'],
        'image_editing_pipeline': ['ImageEditingPipeline'],
        'image_color_enhance_pipeline': ['ImageColorEnhancePipeline'],
        'image_colorization_pipeline': ['ImageColorizationPipeline'],
        'image_instance_segmentation_pipeline':
        ['ImageInstanceSegmentationPipeline'],
        'image_matting_pipeline': ['ImageMattingPipeline'],
        'image_portrait_enhancement_pipeline':
        ['ImagePortraitEnhancementPipeline'],
        'image_reid_person_pipeline': ['ImageReidPersonPipeline'],
        'image_semantic_segmentation_pipeline':
        ['ImageSemanticSegmentationPipeline'],
        'image_style_transfer_pipeline': ['ImageStyleTransferPipeline'],
        'image_super_resolution_pipeline': ['ImageSuperResolutionPipeline'],
        'image_super_resolution_pasd_pipeline':
        ['ImageSuperResolutionPASDPipeline'],
        'image_to_image_translation_pipeline':
        ['Image2ImageTranslationPipeline'],
        'product_retrieval_embedding_pipeline':
        ['ProductRetrievalEmbeddingPipeline'],
        'live_category_pipeline': ['LiveCategoryPipeline'],
        'image_to_image_generate_pipeline': ['Image2ImageGenerationPipeline'],
        'image_to_3d_pipeline': ['Image23DPipeline'],
        'image_inpainting_pipeline': ['ImageInpaintingPipeline'],
        'image_paintbyexample_pipeline': ['ImagePaintbyexamplePipeline'],
        'ocr_detection_pipeline': ['OCRDetectionPipeline'],
        'ocr_recognition_pipeline': ['OCRRecognitionPipeline'],
        'license_plate_detection_pipeline': ['LicensePlateDetectionPipeline'],
        'card_detection_correction_pipeline':
        ['CardDetectionCorrectionPipeline'],
        'table_recognition_pipeline': ['TableRecognitionPipeline'],
        'skin_retouching_pipeline': ['SkinRetouchingPipeline'],
        'face_reconstruction_pipeline': ['FaceReconstructionPipeline'],
        'tinynas_classification_pipeline': ['TinynasClassificationPipeline'],
        'video_category_pipeline': ['VideoCategoryPipeline'],
        'virtual_try_on_pipeline': ['VirtualTryonPipeline'],
        'shop_segmentation_pipleline': ['ShopSegmentationPipeline'],
        'text_driven_segmentation_pipleline':
        ['TextDrivenSegmentationPipeline'],
        'movie_scene_segmentation_pipeline':
        ['MovieSceneSegmentationPipeline'],
        'mog_face_detection_pipeline': ['MogFaceDetectionPipeline'],
        'ulfd_face_detection_pipeline': ['UlfdFaceDetectionPipeline'],
        'retina_face_detection_pipeline': ['RetinaFaceDetectionPipeline'],
        'facial_expression_recognition_pipeline':
        ['FacialExpressionRecognitionPipeline'],
        'facial_landmark_confidence_pipeline':
        ['FacialLandmarkConfidencePipeline'],
        'face_processing_base_pipeline': ['FaceProcessingBasePipeline'],
        'face_attribute_recognition_pipeline': [
            'FaceAttributeRecognitionPipeline'
        ],
        'mtcnn_face_detection_pipeline': ['MtcnnFaceDetectionPipeline'],
        'hand_static_pipeline': ['HandStaticPipeline'],
        'referring_video_object_segmentation_pipeline': [
            'ReferringVideoObjectSegmentationPipeline'
        ],
        'language_guided_video_summarization_pipeline': [
            'LanguageGuidedVideoSummarizationPipeline'
        ],
        'vision_efficient_tuning_adapter_pipeline': [
            'VisionEfficientTuningAdapterPipeline'
        ],
        'vision_efficient_tuning_prompt_pipeline': [
            'VisionEfficientTuningPromptPipeline'
        ],
        'vision_efficient_tuning_prefix_pipeline': [
            'VisionEfficientTuningPrefixPipeline'
        ],
        'vision_efficient_tuning_lora_pipeline': [
            'VisionEfficientTuningLoRAPipeline'
        ],
        'vision_middleware_pipeline': ['VisionMiddlewarePipeline'],
        'vidt_pipeline': ['VidtPipeline'],
        'video_frame_interpolation_pipeline': [
            'VideoFrameInterpolationPipeline'
        ],
        'image_skychange_pipeline': ['ImageSkychangePipeline'],
        'image_driving_perception_pipeline': [
            'ImageDrivingPerceptionPipeline'
        ],
        'vop_retrieval_pipeline': ['VopRetrievalPipeline'],
        'vop_retrieval_se_pipeline': ['VopRetrievalSEPipeline'],
        'video_object_segmentation_pipeline': [
            'VideoObjectSegmentationPipeline'
        ],
        'video_deinterlace_pipeline': ['VideoDeinterlacePipeline'],
        'image_matching_pipeline': ['ImageMatchingPipeline'],
        'image_matching_fast_pipeline': ['ImageMatchingFastPipeline'],
        'video_stabilization_pipeline': ['VideoStabilizationPipeline'],
        'video_super_resolution_pipeline': ['VideoSuperResolutionPipeline'],
        'pointcloud_sceneflow_estimation_pipeline': [
            'PointCloudSceneFlowEstimationPipeline'
        ],
        'face_liveness_ir_pipeline': ['FaceLivenessIrPipeline'],
        'maskdino_instance_segmentation_pipeline': [
            'MaskDINOInstanceSegmentationPipeline'
        ],
        'image_mvs_depth_estimation_pipeline': [
            'ImageMultiViewDepthEstimationPipeline'
        ],
        'ddcolor_image_colorization_pipeline': [
            'DDColorImageColorizationPipeline'
        ],
        'image_structured_model_probing_pipeline': [
            'ImageSturcturedModelProbingPipeline'
        ],
        'video_colorization_pipeline': ['VideoColorizationPipeline'],
        'image_defrcn_fewshot_pipeline': ['ImageDefrcnDetectionPipeline'],
        'image_quality_assessment_degradation_pipeline': [
            'ImageQualityAssessmentDegradationPipeline'
        ],
        'image_open_vocabulary_detection_pipeline': [
            'ImageOpenVocabularyDetectionPipeline'
        ],
        'object_detection_3d_pipeline': ['ObjectDetection3DPipeline'],
        'image_inpainting_sdv2_pipeline': ['ImageInpaintingSDV2Pipeline'],
        'image_quality_assessment_mos_pipeline': [
            'ImageQualityAssessmentMosPipeline'
        ],
        'image_quality_assessment_man_pipeline': [
            'ImageQualityAssessmentMANPipeline'
        ],
        'mobile_image_super_resolution_pipeline': [
            'MobileImageSuperResolutionPipeline'
        ],
        'bad_image_detecting_pipeline': ['BadImageDetecingPipeline'],
        'image_human_parsing_pipeline': ['ImageHumanParsingPipeline'],
        'nerf_recon_acc_pipeline': ['NeRFReconAccPipeline'],
        'nerf_recon_4k_pipeline': ['NeRFRecon4KPipeline'],
        'nerf_recon_img_to_mv_pipeline': ['NeRFReconImgToMVPipeline'],
        'surface_recon_common_pipeline': ['SurfaceReconCommonPipeline'],
        'controllable_image_generation_pipeline': [
            'ControllableImageGenerationPipeline'
        ],
        'image_bts_depth_estimation_pipeline': [
            'ImageBTSDepthEstimationPipeline'
        ],
        'pedestrian_attribute_recognition_pipeline': [
            'PedestrainAttributeRecognitionPipeline'
        ],
        'image_panoptic_segmentation_pipeline': [
            'ImagePanopticSegmentationPipeline',
        ],
        'text_to_360panorama_image_pipeline': [
            'Text2360PanoramaImagePipeline'
        ],
        'human3d_render_pipeline': ['Human3DRenderPipeline'],
        'human3d_animation_pipeline': ['Human3DAnimationPipeline'],
        'image_local_feature_matching_pipeline': [
            'ImageLocalFeatureMatchingPipeline'
        ],
        'rife_video_frame_interpolation_pipeline': [
            'RIFEVideoFrameInterpolationPipeline'
        ],
        'anydoor_pipeline': ['AnydoorPipeline'],
        'image_depth_estimation_marigold_pipeline': [
            'ImageDepthEstimationMarigoldPipeline'
        ],
        'self_supervised_depth_completion_pipeline': [
            'SelfSupervisedDepthCompletionPipeline'
        ],
        'human_normal_estimation_pipeline': ['HumanNormalEstimationPipeline'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
