require 'torch'
require 'nn'
require 'image'

ros = require 'ros'

ros.init('image_captioning')


require 'densecap.DenseCapModel'
local utils = require 'densecap.utils'
local box_utils = require 'densecap.box_utils'
local vis_utils = require 'densecap.vis_utils'


--[[
Run a trained DenseCap model on images.

The inputs can be any one of:
- a single image: use the flag '-input_image' to give path
- a directory with images: use flag '-input_dir' to give dir path
- MSCOCO split: use flag '-input_split' to identify the split (train|val|test)

The output can be controlled with:
- max_images: maximum number of images to process. Set to -1 to process all
- output_dir: use this flag to identify directory to write outputs to
- output_vis: set to 1 to output images/json to the vis directory for nice viewing in JS/HTML
--]]


local cmd = torch.CmdLine()

-- Model options
cmd:option('-checkpoint',
  'data/models/densecap/densecap-pretrained-vgg16.t7')
cmd:option('-image_size', 720)
cmd:option('-rpn_nms_thresh', 0.7)
cmd:option('-final_nms_thresh', 0.3)
cmd:option('-num_proposals', 1000)

-- Input settings
cmd:option('-input_image', '',
  'A path to a single specific image to caption')
cmd:option('-input_dir', '', 'A path to a directory with images to caption')
cmd:option('-input_split', '',
  'A VisualGenome split identifier to process (train|val|test)')

-- Only used when input_split is given
cmd:option('-splits_json', 'info/densecap_splits.json')
cmd:option('-vg_img_root_dir', '', 'root directory for vg images')

-- Output settings
cmd:option('-max_images', 100, 'max number of images to process')
cmd:option('-output_dir', '')
    -- these settings are only used if output_dir is not empty
    cmd:option('-num_to_draw', 10, 'max number of predictions per image')
    cmd:option('-text_size', 2, '2 looks best I think')
    cmd:option('-box_width', 2, 'width of rendered box')
cmd:option('-output_vis', 1,
  'if 1 then writes files needed for pretty vis into vis/ ')
cmd:option('-output_vis_dir', 'vis/data')

-- Misc
cmd:option('-gpu', 0)
cmd:option('-use_cudnn', 1)
local opt = cmd:parse(arg)


function run_image(model, img_in, opt, dtype)

  -- Load, resize, and preprocess image
  img = image.scale(img_in, opt.image_size):float()
  local H, W = img:size(2), img:size(3)
  local img_caffe = img:view(1, 3, H, W)
  img_caffe = img_caffe:index(2, torch.LongTensor{3, 2, 1}):mul(255)
  local vgg_mean = torch.FloatTensor{103.939, 116.779, 123.68}
  vgg_mean = vgg_mean:view(1, 3, 1, 1):expand(1, 3, H, W)
  img_caffe:add(-1, vgg_mean)

  -- Run the model forward
  local boxes, scores, captions = model:forward_test(img_caffe:type(dtype))
  local boxes_xywh = box_utils.xcycwh_to_xywh(boxes)

  local out = {
    img = img,
    boxes = boxes_xywh,
    scores = scores,
    captions = captions,
  }
  return out
end

function result_to_json(result)
  local out = {}
  out.boxes = result.boxes:float():totable()
  out.scores = result.scores:float():view(-1):totable()
  out.captions = result.captions
  return out
end

function lua_render_result(result, opt)
  -- use lua utilities to render results onto the image (without going)
  -- through the vis utilities written in JS/HTML. Kind of ugly output.

  -- respect the num_to_draw setting and slice the results appropriately
  local boxes = result.boxes
  local num_boxes = math.min(opt.num_to_draw, boxes:size(1))
  boxes = boxes[{{1, num_boxes}}]
  local captions_sliced = {}
  for i = 1, num_boxes do
    table.insert(captions_sliced, result.captions[i])
  end

  -- Convert boxes and draw output image
  local draw_opt = { text_size = opt.text_size, box_width = opt.box_width }
  local img_out = vis_utils.densecap_draw(result.img, boxes, captions_sliced, draw_opt)
  return img_out
end


-- Load the model, and cast to the right type
local dtype, use_cudnn = utils.setup_gpus(opt.gpu, opt.use_cudnn)
local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model
model:convert(dtype, use_cudnn)
model:setTestArgs{
  rpn_nms_thresh = opt.rpn_nms_thresh,
  final_nms_thresh = opt.final_nms_thresh,
  num_proposals = opt.num_proposals,
}
model:evaluate()

--TODO make below codes as callback function of subscriber!
--TODO publish ros message instead of writing json


spinner = ros.AsyncSpinner()
spinner:start()

nodehandle = ros.NodeHandle()

subscriber = nodehandle:subscribe("captioning_request", 'sensor_msgs/Image', 1, { 'udp', 'tcp' }, { tcp_nodelay = true })

string_spec = ros.MsgSpec('std_msgs/String')
publisher = nodehandle:advertise("captioning_result", string_spec, 1, false)
ros_msg = ros.Message(string_spec)

subscriber:registerCallback(
function(msg, header)
print("got request")
-- make image from ros msg
local h = msg.height
local w = msg.width
local en = msg.encoding
local data = msg.data

msg_img = data:reshape(h,w,3)
msg_img = msg_img:float()
msg_img = msg_img:mul(1.0/255.0)
msg_img = msg_img:permute(3,1,2)

-- get paths to all images we should be evaluating

local result = run_image(model, msg_img, opt, dtype) 

local ros_boxes = result.boxes:float():totable() --xywh
local ros_scores = result.scores:float():view(-1):totable()
local ros_captions = result.captions
local ros_boxes_string = ""
local ros_scores_string = ""
local ros_captions_string = ""

print(#ros_boxes)

num_of_captions = #ros_boxes
for k=1,#ros_boxes do
ros_boxes_string = ros_boxes_string .. tostring(ros_boxes[k][1]) .. ' ' .. tostring(ros_boxes[k][2]) .. ' ' .. tostring(ros_boxes[k][3]) .. ' ' .. tostring(ros_boxes[k][4]) .. '\n'
ros_scores_string = ros_scores_string .. tostring(ros_scores[k]) .. '\n'
ros_captions_string = ros_captions_string .. ros_captions[k] .. '\n'
end 

--local img_out = lua_render_result(result, opt)
--image.save('dense_cap_test.jpg', img_out)
ros_msg_data = tostring(num_of_captions) .. '\n' .. ros_boxes_string .. '\n' .. ros_scores_string .. '\n' .. ros_captions_string
ros_msg.data = ros_msg_data

ros_msg.header = header

publisher:publish(ros_msg)


end
)


while ros.ok() do
	ros.spinOnce()
	sys.sleep(0.1)
end

subscriber:shutdown()
ros.shutdown()



