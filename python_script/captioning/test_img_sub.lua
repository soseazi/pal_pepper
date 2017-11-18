require 'image'
ros = require 'ros'

ros.init('testimgsub')


spinner = ros.AsyncSpinner()
spinner:start()

nodehandle = ros.NodeHandle()

subscriber = nodehandle:subscribe("/pepper_robot/camera/front/image_raw", 'sensor_msgs/Image', 100, { 'udp', 'tcp' }, { tcp_nodelay = true })


subscriber:registerCallback(
function(msg, header)
-- make image from ros msg
local h = msg.height
local w = msg.width
local en = msg.encoding
local data = msg.data

img = data:reshape(h,w,3)
img = img:float()
img = img:mul(1.0/255.0)
img = img:permute(3,1,2)

image.save('testimgsub.jpg',img)
end
)

for k=1,10 do
	ros.spinOnce()
	sys.sleep(1)
end

ros.shutdown()


