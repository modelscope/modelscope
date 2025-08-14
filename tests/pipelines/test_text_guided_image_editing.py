import unittest
from unittest.mock import MagicMock, patch
from PIL import Image
import numpy as np
import torch

from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys
from modelscope.utils.test_utils import test_level

class TestTextGuidedImageEditing(unittest.TestCase):
    def setUp(self):
        self.task = 'text-guided-image-editing'
        self.model_id = 'AI-ModelScope/instruct-pix2pix'
        
    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    @patch('diffusers.StableDiffusionInstructPix2PixPipeline')
    def test_simple_editing(self, mock_pipe):
        """Test basic editing functionality"""
        # Create mock object
        mock_pipe_instance = mock_pipe.return_value
        mock_image = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
        mock_pipe_instance.return_value.images = [mock_image]
        
        # Create test image
        test_image = Image.new('RGB', (512, 512), color='blue')
        
        # Initialize pipeline
        editor = pipeline(self.task, model=self.model_id)
        editor.pipe = mock_pipe_instance  # Inject mock object
        
        # Perform editing
        result = editor(
            image=test_image,
            instruction='Add a sun',
            style_prompt='Cartoon style'
        )
        
        # Validate results
        self.assertIn(OutputKeys.OUTPUT_IMAGE, result)
        self.assertIsInstance(result[OutputKeys.OUTPUT_IMAGE], Image.Image)
        self.assertEqual(result[OutputKeys.OUTPUT_IMAGE].size, (512, 512))
        
        # Verify call parameters
        mock_pipe_instance.assert_called_once()
        call_args = mock_pipe_instance.call_args[1]
        self.assertEqual(call_args['prompt'], 'Add a sun, Cartoon style')
        self.assertEqual(call_args['num_inference_steps'], 20)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_error_handling(self):
        """Test handling of invalid inputs"""
        editor = pipeline(self.task, model=self.model_id)
        
        # Test missing instruction
        with self.assertRaises(ValueError):
            editor(image='test.jpg')
        
        # Test invalid image path
        with self.assertRaises(ValueError):
            editor(
                image='non_existent.jpg',
                instruction='Test instruction'
            )

if __name__ == '__main__':
    unittest.main()