 local hand_cfg = {
  class_count = 1,  -- excluding background class
  target_smaller_side = 480,
  scales = { 45, 55, 65, 75, 85, 95 },
  max_pixel_size = 1000,
  normalization = { method = 'contrastive', width = 5, centering = true, scaling = true },
--  normalization = { method = 'debug', width = 5, centering = false, scaling = false},
  --augmentation = { vflip = 0.5, hflip = 0.5, random_scaling = 0.0, aspect_jitter = 0.0 },
  augmentation = { vflip = 0, hflip = 0, random_scaling = 0.0, aspect_jitter = 0.0 },
  color_space = 'rgb',
  roi_pooling = { kw = 6, kh = 6 },
  examples_base_path = '',
  background_base_path = '',
  batch_size = 256,
  positive_threshold = 0.7, 
  negative_threshold = 0.3,
  best_match = true,
  test_logger = true,
  nearby_aversion = true,
  use_stn = true,
  vis_stn = true
}

return hand_cfg
