# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

### CONFIGURACION A ENTRENAR:
configuracion = "cfg_ps_6_v2"


import numpy as np
import pandas as pd
import importlib
import sys
import random
from tqdm import tqdm
import gc
import argparse
import torch
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from collections import defaultdict
# import cv2
from copy import copy
import os
#from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import SequentialSampler, DataLoader


# cv2.setNumThreads(0) #OpenCV (open computer vision)   -   No usado

sys.path.append("configs") # Se añaden carpetas al intérprete de python para que cuando busque módulos, busque tambien es estos directorios
sys.path.append("models")  # Fuente: https://www.geeksforgeeks.org/sys-path-in-python/
sys.path.append("data")
# sys.path.append("losses")    -    No usado
# sys.path.append("utils")    -    No usado
os.environ["PATH"] += os.pathsep + r"C:\Users\pepe\anaconda3\envs\Pajaros\Library\bin"


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# Fuentes:
#    https://pytorch.org/docs/stable/data.html
#    https://androidkt.com/create-dataloader-with-collate_fn-for-variable-length-input-in-pytorch/
#
def get_train_dataloader(train_ds, cfg): # Crea el Dataloader para el Dataset y
    train_dataloader = DataLoader(
        train_ds,
        sampler=None,
        shuffle=True,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=False,
        collate_fn=cfg.tr_collate_fn, # collate_fn -> Internally, PyTorch uses a Collate Function to combine the data in your batches together. By
# default, a function called default_collate checks what type of data your Dataset returns and tries to combine into a batch like (x_batch, y_batch).
        drop_last=cfg.drop_last, # a True, desecha el último batch si no es completo, es decir si el Train no es divisible exacto por el batch_size
        worker_init_fn=worker_init_fn, # No entiendo como llama a esta clase
        # Workers  -  whow many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: 0)
    )
    print(f"train: dataset {len(train_ds)}, dataloader {len(train_dataloader)}")
    return train_dataloader

# Se crea el dataloader para la validación
#     Se crea un "sampler" que es un objeto que simplemente va pasando los datos del dataset de uno en uno, siempre en el mismo orden
def get_val_dataloader(val_ds, cfg):
    sampler = SequentialSampler(val_ds)
    if cfg.batch_size_val is not None:
        batch_size = cfg.batch_size_val
    else:
        batch_size = cfg.batch_size
        #Dataloader -> (val_set, sampler, batch_size, num_workers, collate_fn, worker_init_fn)
    val_dataloader = DataLoader(
        val_ds,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        pin_memory=False, # si True -> copia tensores en memoria pinned de cuda ...
        collate_fn=cfg.val_collate_fn,
        worker_init_fn=worker_init_fn,
    )
    print(f"valid: dataset {len(val_ds)}, dataloader {len(val_dataloader)}")
    return val_dataloader

# Fuentes: https://huggingface.co/docs/transformers/main_classes/optimizer_schedules
def get_scheduler(cfg, optimizer, total_steps):
    # scheduler -> Ajuste dinámico del Learning Rate del optimizador.  Se actualiza por el nº de epoch
    # get_cosine_schedule_with_warmup  ->  tasa de aprendizaje que disminuye siguiendo un crecimiento inicial y una caida prolongada
    # optimizer -> All optimization logic is encapsulated in the optimizer object. Here, we use the SGD optimizer; additionally,
    #              there are many different optimizers available in PyTorch such as ADAM and RMSProp, that work better for different kinds of models
    #              and data.
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup * (total_steps // cfg.batch_size),
        num_training_steps=cfg.epochs * (total_steps // cfg.batch_size),
    )
    return scheduler

# Fuentes:
#   random.seed - w3schools.com/python/ref_random_seed.asp
def set_seed(seed=1234): # Poned seed a varias cosas
    random.seed(seed) # pone seed a la funcion random
    os.environ["PYTHONHASHSEED"] = str(seed) # semilla a la funcion hash
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

#
#   de un archivo de config aleatorio -> "cfg.model = "ps_model_11" "
def get_model(cfg, train_dataset):
    Net = importlib.import_module(cfg.model).Net #Crea un objeto de la clase Net (nn.model), perteneciente al módulo cfg.model.
    return Net(cfg)                              # Es decir, carga en un objeto una de sus redes de estructura personalizada.

# Crea un objeto "checkpoint" en el que se guarda los "objeto map de python" de el modelo y el optimizador + el nº de epoch
#       - el modelo, obj map entre los parámetros y las capas de la red
#     objeto .map() ->  Es una funcion que pasa una lista, de forma iterativa por una función.
# Ejemplo:   x = [1 ,2 ,3 ,4]  res = objMap(elevarCuadrado(), x)   res = [1, 4, 9, 16]
# Fuente: https://programmerclick.com/article/5104512636/
def create_checkpoint(model, optimizer, epoch, scheduler=None, scaler=None):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    return checkpoint


# Copiamos el obj de configuracion puesto a mano
cfg = copy(importlib.import_module(configuracion).cfg)


os.makedirs(str(cfg.output_dir + "/"), exist_ok=True)  # Crea el directorio si no existe, de lo contrario lo deja igual.
# Fuente: https://www.geeksforgeeks.org/python-os-makedirs-method/

cfg.CustomDataset = importlib.import_module(cfg.dataset).CustomDataset  # Guarda la clase del Dataset personalizado CustomDataset(Dataset)
cfg.tr_collate_fn = importlib.import_module(cfg.dataset).tr_collate_fn  # train collate -> funcion que hace los batch o cjtos de X e y
cfg.val_collate_fn = importlib.import_module(cfg.dataset).val_collate_fn  # validation collate lo mismo \ En todos los modulos data ambas = None
batch_to_device = importlib.import_module(cfg.dataset).batch_to_device  # Guarda la funcion batch_to_device(batch, device)
# device -> La GPU de cfg.     La func hace : "batch_dict = {key: batch[key].to(device) for key in batch}"
# Básicamente usa .to() que es para pasarselo a la GPU


# Fuentes:
#   nn.Module.eval()  -  https://stackoverflow.com/questions/48146926/whats-the-meaning-of-function-eval-in-torch-nn-module
#   collections.defaultdict  -  https://stackoverflow.com/questions/5900578/how-does-collections-defaultdict-work
#   Automatic Mixed Precision Package  -  https://pytorch.org/docs/stable/amp.html
#   Iterating Through .items()  -  https://realpython.com/iterate-through-dictionary-python/
#   torch.cat vs torch.stack  -  https://machinelearningknowledge.ai/pytorch-stack-vs-cat-explained-for-beginners/
def run_eval(model, val_dataloader, cfg, pre="val"):
    # CONFIGURAMOS EL MODELO EN EVALUACION
    model.eval()  # Dropout and BatchNorm se comportan distinto Durante el entrenamiento y la evaluación. Debes avisar a la red de que va a realizar
    # This sets self.training to False for every module in the model
    torch.set_grad_enabled(
        False)  # No queremos entrenar -> no hacemos backpropagation Por tanto no tiene sentido ir computando los grafos de derivadas

    val_data = defaultdict(
        list)  # defaultdict creará cualquier objeto de un diccionario al que intentes acceder y no esté contenido en el mismo
    # default items are created using int(), which will return the integer object 0. For the second example, default items are created using list(), which returns a new empty list object.

    # BUCLE QUE ITERA POR TODOS LOS BATCH
    for data in tqdm(val_dataloader):  # tqdm -> barra de progreso
        # val_dataloader es un conjunto de grupos de datos, un conjunto de batch -> data es una iteracion de los batch

        batch = batch_to_device(data, device)

        # FORWARD PROP  (CON O SIN PERMITIR PRECISION MEZCLADA)
        if cfg.mixed_precision:
            with autocast():  # Permite precision mezaclada  -> esto es, algunas partes del modelo en float16 y otras en float32 para ser mas eficiente
                output = model(batch)  # Lo permite solo durante el forward propagation
        else:
            output = model(batch)

        #    ¿¿¿¿????     ¿Guarda en el nuevo diccionario val_data la nueva key y le asigna o suma el valor?
        for key, val in output.items():
            val_data[key] += [output[key]]

    # ¿¿¿???   ???¿¿¿
    for key, val in output.items():
        value = val_data[key]
        if len(value[0].shape) == 0:
            val_data[key] = torch.stack(
                value)  # si hay 0 elementos en el array valor de la clave, concatena los tensores en la dimension existente
        else:
            val_data[key] = torch.cat(value,
                                      dim=0)  # si no, los concatena a lo largo de una nueva dimensión (un vector sobre otro, haciendo una mtx)

    if cfg.save_val_data:
        torch.save(val_data, f"{cfg.output_dir}/{pre}_data_seed{cfg.seed}.pth")

    if "loss" in val_data:  # NO ENTIENDO, ¿¿¿NO DEBERÍA HABER SIEMPRE LOSS???
        val_losses = val_data["loss"].cpu().numpy()  # Convierte el tensor en numpy con la cpu
        val_loss = np.mean(val_losses)
        print(f"Mean {pre}_loss", np.mean(val_losses))

    else:
        val_loss = 0.0

    print("EVAL FINISHED")

    return val_loss


if __name__ == "__main__":

    if cfg.seed < 0:
        cfg.seed = np.random.randint(1_000_000)
    print("seed", cfg.seed)

    # Sustituyo el original por cpu
    # device = "cuda:%d" % cfg.gpu
    device = "cpu"
    cfg.device = device

    set_seed(cfg.seed)  # Llama a la funcion creada que establece varias seed para la generacion aleatoria

    train_df = pd.read_csv(cfg.train_df)
    val_df = pd.read_csv(cfg.val_df)

    # Custom Dataset - https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    # El obj DataSet tiene el método "getItem", que devuelve un Sample. Normalmente un par (imagen, lable) pedido con un índice iterado el dataloader).
    # En este caso devuelve un Diccionario -> feature_dict = {"input":tensor, "target":tensor, "weight":tensor, "fold":tensor}
    #    input -> wav_tensor (fragmento de audio original, con offset aleatorio. Data Aug si lo indica la config. tanto para "train" como "val")
    #    target -> label hecha tensor
    #    weight ->
    #    fold ->
    train_dataset = cfg.CustomDataset(train_df, cfg, aug=cfg.train_aug, mode="train")
    val_dataset = cfg.CustomDataset(val_df, cfg, aug=cfg.val_aug, mode="val")

    train_dataloader = get_train_dataloader(train_dataset, cfg)
    val_dataloader = get_val_dataloader(val_dataset, cfg)

    model = get_model(cfg, train_dataset)
    model.to(device)

    total_steps = len(train_dataset)

    params = model.parameters()
    optimizer = optim.Adam(params, lr=cfg.lr, weight_decay=0)

    scheduler = get_scheduler(cfg, optimizer,
                              total_steps)  # Ajuste dinámico del Learning Rate del optimizador. Se ajusta cada Epoch

    if cfg.mixed_precision:  # Fuente - https://pytorch.org/docs/stable/amp.html#gradient-scaling
        scaler = GradScaler()  # Obj escala los gradientes para evitar que grad demasiado pequeños se pierdan por no poder ser representados en float16
    else:
        scaler = None

    cfg.curr_step = 0
    i = 0  # ¿NO SE USA?
    best_val_loss = np.inf  # Mejor pérdida = "pérdida menor" | Se pone a infinito para empezar  ¿¿¿NO SE USA???
    optimizer.zero_grad()  # Se vacían los gradientes que pudieran quedar

    # BUCLE:
    for epoch in range(cfg.epochs):

        set_seed(cfg.seed + epoch)  # Se cambia la seed cada epoch

        cfg.curr_epoch = epoch

        print("EPOCH:", epoch)

        progress_bar = tqdm(range(len(train_dataloader)))  # Barra de progreso

        tr_it = iter(train_dataloader)  # Train Iterator del DataLoader

        losses = []

        gc.collect()  # ¿Libera memoria? ¿¿"Elimina objetos no utilizados o inalcanzables"??

        if cfg.train:
            # ==== TRAIN LOOP
            for itr in progress_bar:  # Bucle que pasa por todos los batch que componen un epoch
                i += 1  # ¿¿¿No se usa???

                cfg.curr_step += cfg.batch_size  # Se lleva la cuenta del nº de sample por el que vamos

                data = next(tr_it)  # data = siguiente batch del dataloader

                model.train()  # Avisas a tu modelo de que vas a entrenar, para que use las capas de dropout y otras que no se usan al hacer Validacion.
                torch.set_grad_enabled(True)

                batch = batch_to_device(data, device)  # Se carga el batch

                if cfg.mixed_precision:  # Forward con o sin Mixed precision
                    with autocast():
                        output_dict = model(batch)
                else:
                    output_dict = model(batch)

                loss = output_dict["loss"]

                losses.append(loss.item())  # Se añade a la lista de pérdidas

                if cfg.mixed_precision:
                    scaler.scale(
                        loss).backward()  # Se reescalan los ajustes de los parámetros para evitar UnderFlow por ser float16
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                else:
                    loss.backward()  # Se actualizan los derivadas en las capas
                    optimizer.step()  # Se aplica la actualizacion de los parámetros
                    optimizer.zero_grad()  # Se vacia el buffer

                if scheduler is not None:
                    scheduler.step()  # Actualiza Learning Rate

                if cfg.curr_step % cfg.batch_size == 0:  # ¿¿¿No sucede siempre???
                    progress_bar.set_description(
                        f"loss: {np.mean(losses[-10:]):.4f}")  # Se imprime la media de las perdidas de los últimos 10 batch

        if cfg.val:
            if (epoch + 1) % cfg.eval_epochs == 0 or (
                    epoch + 1) == cfg.epochs:  # ¿Primera mitad? la segunda mitad siempre se activa en la ultima vuelta del bucle.
                val_loss = run_eval(model, val_dataloader, cfg)
            else:
                val_score = 0

        if cfg.epochs > 0:  # Se guarda el modelo de la ultima epoch
            checkpoint = create_checkpoint(
                model, optimizer, epoch, scheduler=scheduler, scaler=scaler
            )

            torch.save(checkpoint, f"{cfg.output_dir}/checkpoint_last_seed{cfg.seed}.pth")

    if cfg.epochs > 0:
        checkpoint = create_checkpoint(model, optimizer, epoch, scheduler=scheduler, scaler=scaler)
        # guarda os parámetros en la direccion de config + seed   \   Los guarda en Output dir especificada en Default Config
        torch.save(checkpoint, f"{cfg.output_dir}/checkpoint_last_seed{cfg.seed}.pth")