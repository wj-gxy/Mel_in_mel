import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict, build_env
from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
from model import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss, \
    discriminator_loss
from model_6 import stego_model
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint
import swanlab
torch.backends.cudnn.benchmark = True

def train(rank, a, h):


    swanlab.init(
        project='Mel_in',
        config={
            "learning_rate": h.learning_rate,
            "batch_size": h.batch_size,
            "epochs": a.training_epochs,
            "architecture": "HiFiGAN+Stego"
        },
        resume=True,
        name=f"exp_{time.strftime('%Y%m%d-%H%M%S')}"  # 实验名称（带时间戳）
    )

    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    generator = Generator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    model = stego_model().to(device)
    if rank == 0:
        # print(generator)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        print("c z")
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_0')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_0')

    steps = 0
    if cp_g is None or cp_do is None:
        print("no")
        state_dict_do = None
        last_epoch = -1
    else:
        print("!!!!!!!!!!!!!!!!")
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)

        generator.load_state_dict(state_dict_g['generator'])
        model.load_state_dict(state_dict_g['model'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])

        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank], find_unused_parameters=True).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank], find_unused_parameters=True).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank], find_unused_parameters=True).to(device)
        model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True).to(device)

    optim_g = torch.optim.AdamW(itertools.chain(generator.parameters(), model.parameters()),
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

        optim_g.param_groups[0]['capturable'] = True
        optim_d.param_groups[0]['capturable'] = True

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_files, validation_files = get_dataset_filelist(a)
    trainset = MelDataset(training_files, h.segment_size, h.n_fft, h.num_mels,
                          h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                          shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax_for_loss, device=device,
                          fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir)
    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler, batch_size=h.batch_size, pin_memory=True, drop_last=True)

    if rank == 0:
        validset = MelDataset(validation_files, h.segment_size, h.n_fft, h.num_mels,
                              h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, True, True, n_cache_reuse=0,
                              fmax_loss=h.fmax_for_loss, device=device, fine_tuning=a.fine_tuning,
                              base_mels_path=a.input_mels_dir)

        validation_loader = DataLoader(validset, num_workers=0, shuffle=False,
                                       sampler=None, batch_size=1, pin_memory=True, drop_last=True)



    model.train()
    generator.train()
    mpd.train()
    msd.train()

    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):

            if rank == 0:
                start_b = time.time()
            x_mel, x_aud, x_mel_loss, y_mel, y_aud, y_mel_s = batch

            x_mel = torch.autograd.Variable(x_mel.to(device, non_blocking=True))
            x_aud = torch.autograd.Variable(x_aud.to(device, non_blocking=True))
            x_aud = x_aud.unsqueeze(1)

            x_mel_loss = torch.autograd.Variable(x_mel_loss.to(device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            y_mel_s = torch.autograd.Variable(y_mel_s.to(device, non_blocking=True))

            y_g_hat = generator(x_mel, y_mel)  # 生成语音


            y_g_hat_mel_loss = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size,
                                               h.win_size, h.fmin, h.fmax_for_loss)

            re_serect_mel = model(y_g_hat_mel_loss)



            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(x_aud, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(x_aud, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
            loss_disc_all = loss_disc_s + loss_disc_f
            loss_disc_all.backward()
            optim_d.step()

            # L1 Mel-Spectrogram Loss

            optim_g.zero_grad()
            loss_mel = F.l1_loss(x_mel_loss, y_g_hat_mel_loss)
            loss_mel_ser = F.l1_loss(y_mel, re_serect_mel)
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(x_aud, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(x_aud, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel* 45 + loss_mel_ser* 45
            loss_gen_all.backward()
            optim_g.step()
            if rank == 0:
                swanlab.log({
                    "gen_loss_total": loss_gen_all.item(),
                    "mel_loss": loss_mel.item(),  # 原始Mel损失
                    "mel_ser_loss": loss_mel_ser.item(),  # Stego模型的Mel重建损失
                    "disc_loss": loss_disc_all.item(),  # 判别器总损失
                    "gen_loss_f": loss_gen_f.item(),  # 生成器对MPD的损失
                    "gen_loss_s": loss_gen_s.item(),  # 生成器对MSD的损失
                    "feature_loss_f": loss_fm_f.item(),  # MPD特征匹配损失
                    "feature_loss_s": loss_fm_s.item(),  # MSD特征匹配损失
                }, step=steps)

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(x_mel_loss, y_g_hat_mel_loss).item()
                        ser_error = F.l1_loss(y_mel, re_serect_mel).item()

                    print(
                        'Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, Mel-Ser. Error : {:4.3f}, s/b : {:4.3f}'.
                        format(steps, loss_gen_all, mel_error, ser_error, time.time() - start_b))

                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict(),
                                     'model': (model.module if h.num_gpus > 1 else model).state_dict()})

                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'mpd': (mpd.module if h.num_gpus > 1
                                             else mpd).state_dict(),
                                     'msd': (msd.module if h.num_gpus > 1
                                             else msd).state_dict(),
                                     'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                                     'epoch': epoch})


                # Validation
                if steps % a.validation_interval == 0:  # and steps != 0:
                    generator.eval()
                    model.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    val_err_ser = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            x_mel, x_aud, x_mel_loss, y_mel, y_aud, y_mel_s = batch

                            x_mel_loss = torch.autograd.Variable(x_mel_loss.to(device, non_blocking=True))
                            y_mel_s = torch.autograd.Variable(y_mel_s.to(device, non_blocking=True))

                            # print(x_mel.size())
                            # print(y_mel.size())

                            y_g_hat = generator(x_mel, y_mel)
                            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                                          h.hop_size, h.win_size, h.fmin, h.fmax_for_loss)
                            re_serect_mel = model(y_g_hat_mel)

                            val_err_tot += F.l1_loss(x_mel_loss, y_g_hat_mel).item()
                            val_err_ser += F.l1_loss(y_mel_s, re_serect_mel).item()

                            if j <= 4:

                                y_hat_spec = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                                                             h.sampling_rate, h.hop_size, h.win_size,
                                                             h.fmin, h.fmax)

                        val_err = val_err_tot / (j + 1)
                        val_err_ser = val_err_ser / (j + 1)


                    model.train()
                    generator.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_wavs_dir', default='/home/lgd/miniconda3/envs/dataset/LJSpeech-1.1/wavs' )#/home/lgd/miniconda3/envs/dataset/aishell3
    parser.add_argument('--input_mels_dir', default='ft_dataset')
    parser.add_argument('--input_training_file',
                        default='/home/lgd/miniconda3/envs/dataset/LJSpeech-1.1/training.txt')
    parser.add_argument('--input_validation_file',
                        default='/home/lgd/miniconda3/envs/dataset/LJSpeech-1.1/validation.txt')
    parser.add_argument('--checkpoint_path',
                        default='/home/lgd/miniconda3/envs/all/Mel_in/cp_hifigan/hifigan/hifigan_lj/new_lstm')
    parser.add_argument('--config',
                        default='/home/lgd/miniconda3/envs/all/Mel_in/cp_hifigan/hifigan/hifigan_lj/config.json')#/home/lgd/miniconda3/envs/all/Mel_in/cp_hifigan/ashell3_22050/

    parser.add_argument('--training_epochs', default=3100, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        print(h.num_gpus)
        h.batch_size = int(h.batch_size / h.num_gpus)

        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))  # 多进程训练
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()
