from random import choices

crop=transforms.functional.crop
idx=100
ds=ImageFolder(DATASET_PATH+ '/test_images',   transform=transforms.ToTensor())
N=20

fig,all_axs=plt.subplots(N,2,figsize=(20,N*10))

imgs=choices(ds,k=N)

for it,(x,axs) in enumerate(zip(imgs,all_axs)):
    #print(x[0].shape)
    rpn.eval()
    out=rpn(x[0].cuda().unsqueeze(0))
    out=out[0]
    img=x[0]
    #print(out[0])
    box=None
    #print(out)
    if len(out['boxes'])>1:
        for b, label,score in zip(out['boxes'],out['labels'],out['scores']):
            if label==16 and score>0.95:
                box=tuple(int(t) for t in b.cpu())
                print(it, ' has box')
                break
            
    print(img.shape)
    if box:
        print(box)
        #cropped_img=img[:,box[0]:box[2],box[1]:box[3]]
        cropped_img=img[:,box[1]:box[3],box[0]:box[2]]
    else: cropped_img=img
    axs[0].imshow(img.permute(1,2,0))
    axs[1].imshow(cropped_img.permute(1,2,0))