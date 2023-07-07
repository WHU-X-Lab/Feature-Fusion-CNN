import torch
image = 'groupid54diff0x149521.jpg'
image_id = image.split('groupid')[1].split('diff')[0]
print("img_id",image_id)
# n=torch.FloatTensor(3,3,2,3).fill_(1)
# print("before:",n)
# # n[:,0:1,1:2]=0
# # n[:,1:,0:1]=0
# n = torch.mean(n.view(n.size(0), -1), dim=1)
# print("after:",n)