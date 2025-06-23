def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    print(f"[INFO] CUDA_VISIBLE_DEVICES = {args.gpu}")  # ← 追加
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    gpus = [int(i) for i in args.gpu.split(',')]
    if len(gpus) != 1:
        raise KeyError(f"gpu number must be one during evaluating, but got {gpus}")

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    print(f"[INFO] Input size: {input_size}")  # ← 追加

    print("[INFO] Initializing model...")  # ← 追加
    model = Res_Deeplab(num_classes=args.num_classes)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    print(f"[INFO] Loading dataset from: {args.data_dir}")  # ← 追加
    lip_dataset = InferenceDataSet(args.data_dir, 'val', crop_size=input_size, transform=transform)
    num_samples = len(lip_dataset)
    print(f"[INFO] Number of samples: {num_samples}")  # ← 追加

    valloader = data.DataLoader(lip_dataset, batch_size=args.batch_size * len(gpus),
                                shuffle=False, pin_memory=True)

    restore_from = args.restore_from
    print(f"[INFO] Loading model weights from: {restore_from}")  # ← 追加
    state_dict = model.state_dict().copy()
    state_dict_old = torch.load(restore_from)

    print("[INFO] Restoring model parameters...")  # ← 追加
    for key, nkey in zip(state_dict_old.keys(), state_dict.keys()):
        if key != nkey:
            state_dict[key[7:]] = deepcopy(state_dict_old[key])
        else:
            state_dict[key] = deepcopy(state_dict_old[key])

    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()

    print("[INFO] Starting inference...")  # ← 追加
    parsing_preds, scales, centers = valid(model, valloader, input_size, num_samples, len(gpus))

    #=================================================================
    save_dir = args.output_path
    save_lbl_dir = f'{save_dir}/Pred_parsing_results'
    print(f"[INFO] Saving parsing results to: {save_lbl_dir}")  # ← 追加

    if args.vis == 'yes':
        save_vis_dir = f'{save_dir}/Pred_parsing_results_vis'
        if not os.path.exists(save_vis_dir):
            os.makedirs(save_vis_dir)
        print(f"[INFO] Saving visualization to: {save_vis_dir}")  # ← 追加

    palette = get_lip_palette()
    output_parsing = parsing_preds

    for i, im_lbl_name in tqdm.tqdm(enumerate(lip_dataset.files), total=len(lip_dataset.files)):
        image_path = im_lbl_name['img']
        im_name = im_lbl_name['name']
        im_name = im_name.replace('.jpg', '.png')
        save_lbl_path = os.path.join(save_lbl_dir, im_name)
        os.makedirs(os.path.dirname(save_lbl_path), exist_ok=True)

        img = PILImage.open(image_path)
        w, h = img.size
        pred_out = output_parsing[i]
        s = scales[i]
        c = centers[i]
        pred = transform_parsing(pred_out, c, s, w, h, input_size)
        pred_lbl = PILImage.fromarray(pred)
        pred_lbl.save(save_lbl_path)

        if args.vis == 'yes':
            save_vis_path = os.path.join(save_vis_dir, im_name)
            pred_vis = PILImage.fromarray(pred)
            pred_vis.putpalette(palette)
            pred_vis = pred_vis.convert("RGB")
            pred_vis.save(save_vis_path)

    print("[INFO] Inference completed successfully.")  # ← 追加
