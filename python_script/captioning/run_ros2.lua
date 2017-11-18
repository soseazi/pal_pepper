require 'torch'
require 'nn'
require 'image'
require 'math'

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
cmd:option('-image_size', 224)
cmd:option('-rpn_nms_thresh', 0.7)
cmd:option('-final_nms_thresh', 0.3)
cmd:option('-num_proposals', 10)

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

obj_msg_spec = ros.MsgSpec('pal_pepper/objs')
obj_array_msg_spec = ros.MsgSpec('pal_pepper/objs_array')

subscriber = nodehandle:subscribe("objects", 'pal_pepper/objs_array', 1, { 'udp', 'tcp' }, { tcp_nodelay = true })
subscriber2 = nodehandle:subscribe("captioning_request", 'pal_pepper/objs_array', 1, { 'udp', 'tcp' }, { tcp_nodelay = true })

publisher = nodehandle:advertise("objects_w_caption", obj_array_msg_spec, 1, false)
publisher2 = nodehandle:advertise("captioning_result", obj_array_msg_spec, 1, false)

print("Captioning module ready")

subscriber:registerCallback(
function(msg, header)

for i=1 , #msg.objects do
	if msg.objects[i].class_string == 'person' then
		local h = msg.objects[i].cropped.height
		local w = msg.objects[i].cropped.width
		local en = msg.objects[i].cropped.encoding
		local data = msg.objects[i].cropped.data
		local msg_img = data:reshape(h,w,3)
				
		local im2 = msg_img:clone()
		msg_img[{ {}, {}, 1 }]:copy(im2[{ {}, {}, 3}])
		msg_img[{ {}, {}, 3 }]:copy(im2[{ {}, {}, 1}])				
				
		msg_img = msg_img:float()
		msg_img = msg_img:mul(1.0/255.0)
		msg_img = msg_img:permute(3,1,2)
		-- Load, resize, and preprocess image
		local img = image.scale(msg_img, opt.image_size):float()

		local H, W = img:size(2), img:size(3)
		vgg_mean = torch.FloatTensor{103.939, 116.779, 123.68}
		vgg_mean = vgg_mean:view(1, 3, 1, 1):expand(1, 3, H, W)

		local img_caffe = img:view(1,3, H, W)
		img_caffe = img_caffe:index(2, torch.LongTensor{3, 2, 1}):mul(255)


		img_caffe:add(-1, vgg_mean)
	
		boxes,scores,captions = model:forward_test(img_caffe:type(dtype))
				
		flag_name = 0
		flag_cloth = 0
			
		for j=1, #captions do
			splited = string.split(captions[j]," ")
			for k=1, #splited do
				if ("man" == splited[k] or "boy" == splited[k]) and flag_name == 0 then 
					table.insert(msg.objects[i].tags,"man") 
					table.insert(msg.objects[i].tags,"men") 	
					table.insert(msg.objects[i].tags,"boy") 						
					table.insert(msg.objects[i].tags,"boys") 		
					flag_name=1 
				elseif ("woman" == splited[k] or "girl" == splited[k]) and flag_name == 0 then 
					table.insert(msg.objects[i].tags,"woman") 
					table.insert(msg.objects[i].tags,"women") 	
					table.insert(msg.objects[i].tags,"girl") 						
					table.insert(msg.objects[i].tags,"girls") 		
					flag_name=1 
				end
			end
					
			if string.find(captions[j],"blue shirt") ~= nil and flag_cloth == 0 then table.insert(msg.objects[i].tags,"blue") flag_cloth=1  
			elseif string.find(captions[j],"red shirt") ~= nil and flag_cloth == 0 then table.insert(msg.objects[i].tags,"red") flag_cloth=1   
			elseif string.find(captions[j],"white shirt") ~= nil and flag_cloth == 0 then table.insert(msg.objects[i].tags,"white") flag_cloth=1   
			elseif string.find(captions[j],"black shirt") ~= nil and flag_cloth == 0 then table.insert(msg.objects[i].tags,"black") flag_cloth=1   			
			elseif string.find(captions[j],"green shirt") ~= nil and flag_cloth == 0 then table.insert(msg.objects[i].tags,"green") flag_cloth=1  	
			elseif string.find(captions[j],"blue jacket") ~= nil and flag_cloth == 0 then table.insert(msg.objects[i].tags,"red") flag_cloth=1   
			elseif string.find(captions[j],"red jacket") ~= nil and flag_cloth == 0 then table.insert(msg.objects[i].tags,"red") flag_cloth=1   
			elseif string.find(captions[j],"white jacket") ~= nil and flag_cloth == 0 then table.insert(msg.objects[i].tags,"white") flag_cloth=1   
			elseif string.find(captions[j],"black jacket") ~= nil and flag_cloth == 0 then table.insert(msg.objects[i].tags,"black") flag_cloth=1   			
			elseif string.find(captions[j],"green jacket") ~= nil and flag_cloth == 0 then table.insert(msg.objects[i].tags,"green") flag_cloth=1  end
					
			if string.find(captions[j],"sitting") ~= nil then 
				table.insert(msg.objects[i].tags,"sitting") 
				msg.objects[i].isSitting = 1
			end  
		end
	
		msg.objects[i].captions = captions
		print (msg.objects[i].captions)

	end

end
publisher:publish(msg)
print("")		

		
end
)

subscriber2:registerCallback(
function(msg, header)
print ("got request")
msg.objects = {}
local h = msg.scene_rgb.height
local w = msg.scene_rgb.width
local en = msg.scene_rgb.encoding
local data = msg.scene_rgb.data
local msg_img = data:reshape(h,w,3)
		
local im2 = msg_img:clone()
msg_img[{ {}, {}, 1 }]:copy(im2[{ {}, {}, 3}])
msg_img[{ {}, {}, 3 }]:copy(im2[{ {}, {}, 1}])			
		
		
msg_img = msg_img:float()
msg_img = msg_img:mul(1.0/255.0)
msg_img = msg_img:permute(3,1,2)
-- Load, resize, and preprocess image
local img = image.scale(msg_img, 720):float()

local H, W = img:size(2), img:size(3)
vgg_mean = torch.FloatTensor{103.939, 116.779, 123.68}
vgg_mean = vgg_mean:view(1, 3, 1, 1):expand(1, 3, H, W)

local img_caffe = img:view(1,3, H, W)
img_caffe = img_caffe:index(2, torch.LongTensor{3, 2, 1}):mul(255)


img_caffe:add(-1, vgg_mean)

boxes,scores,captions = model:forward_test(img_caffe:type(dtype))

msg.tags = captions

publisher2:publish(msg)
print("")		

end
)

while ros.ok() do
	ros.spinOnce()
	--sys.sleep(0.1)
end

subscriber:shutdown()
ros.shutdown()



