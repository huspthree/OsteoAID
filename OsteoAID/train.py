from ultralytics import YOLO
import os


if __name__ == '__main__':
    # Loading model
    model = YOLO(r'OsteoAID/models/model/OsteoAID.yaml')

    # Training parameter ----------------------------------------------------------------------------------------------
    model.train(
        data=r'my_dataset/data.yaml',
        epochs=500,  # (int) The number of training cycles
        patience=100,  # (int) The number of cycles to wait for no significant improvement for early stop
        batch=4,  # (int) Number of images per batch (-1 for automatic batch processing)
        imgsz=640,  # (int) Enter the size of the image, integer or w, h
        save=True,  # (bool) Save training checkpoints and prediction results
        save_period=-1,  # (int) Save checkpoints every x cycle (disabled if less than 1)
        cache=False,  # (bool) True/ram, disk, or False. Load data using the cache
        device=2,  # (int | str | list, optional) Running devices, such as cuda device=0 or device=0,1,2,3 or device=cpu
        workers=8,  # (int) Number of worker threads for data loading (per DDP process)
        project='runs/train',  # (str, optional) Project name
        name='OsteoAID',  # (str, optional) The name of the experiment, and the results are saved in the 'project/name' directory
        exist_ok=False,  # (bool) Whether to override existing experiments
        pretrained=True,  # (bool | str) Whether to use a pre-trained model (bool), or a model from which weights are loaded (str)
        optimizer='SGD',  # (str) To use the optimizer, select =[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
        verbose=True,  # (bool) Whether to print detailed output
        seed=0,  # (int) Random seeds for repeatability
        deterministic=True,  # (bool)Whether to enable deterministic mode
        single_cls=False,  # (bool) Train multiple classes of data into a single class
        rect=False,  # (bool) If mode='train', rectangle training is performed, and if mode='val', rectangle verification is performed
        cos_lr=False,  # (bool) Use the cosine learning rate scheduler
        close_mosaic=0,  # (int) Disable Mosaic enhancement for the last few cycles
        resume=True,  # (bool) Resume training from the previous checkpoint
        amp=True,  # (bool) Automatic Mixing accuracy (AMP) training, select =[True, False], True run AMP check

        # hyperparameter ----------------------------------------------------------------------------------------------
        lr0=0.01,  # (float) Initial learning rate (e.g., SGD=1E-2, Adam=1E-3)
        lrf=0.01,  # (float) Final learning rate (lr0 * lrf)
        momentum=0.937,  # (float) SGD momentum /Adam beta1
        weight_decay=0.0005,  # (float) Optimizer weight decay 5e-4
        warmup_epochs=3.0,  # (float) Preheating period (fractions available)
        warmup_momentum=0.8,  # (float) Preheating initial momentum
        warmup_bias_lr=0.1,  # (float) Preheat the initial bias learning rate

    )

