import torch
import torchvision.models
import torch.optim as optim
import os
import time

from torch import Tensor
from tqdm import tqdm

import utils.evaluate as evaluate

from loguru import logger
from data.data_loader import sample_dataloader

import modules as mods

def AllLoss(code_length, gamma, alpha, beta):
    def forward(U_latent, U_image, U_label, U_ae_rec, U_ae_target, B, S):
        def feature_loss(F, B, S):
            # hash_loss = ((code_length * S - F @ B.t()) ** 2).mean()
            hash_loss = (((code_length * S - F @ B.t()) * 12.0 / code_length) ** 2).mean()
            quantization_loss = ((F - B) ** 2).mean()
            return hash_loss + gamma * quantization_loss

        def ae_loss(recon_x, x):
            return ((recon_x - x) ** 2).mean()

        loss = (
                       feature_loss(U_image, B, S)
                       + alpha * feature_loss(U_label, B, S)
                       + beta * (feature_loss(U_latent, B, S)
                       + ae_loss(U_ae_rec, U_ae_target))
               )

        return loss

    return forward

def train(
        query_dataloader,
        retrieval_dataloader,
        code_length,
        logdir,
        args
):
    """
    Training model.

    Args
        query_dataloader, retrieval_dataloader(torch.utils.data.dataloader.DataLoader): Data loader.
        code_length(int): Hashing code length.
        args: arguments.


    Returns
        mAP(float): Mean Average Precision.
    """
    num_retrieval = len(retrieval_dataloader.dataset)

    # Initialization
    model_image = mods.alexnet(code_length).to(args.device)
    model_label = mods.mlp(args.class_num, code_length).to(args.device)
    model_gcn = mods.GCN(args.embedding_size, args.hidden_size, code_length, args.gcn_dropout).to(args.device)
    model_ae_image = mods.ae(code_length, args.hidden_size, args.embedding_size).to(args.device)

    criterion_all = AllLoss(code_length, args.gamma, args.alpha, args.beta)

    optimizer_image = optim.Adam(model_image.parameters(), lr=args.lr, weight_decay=1e-5)
    optimizer_label = optim.Adam(model_label.parameters(), lr=args.lr, weight_decay=1e-5)
    optimizer_gcn = optim.Adam(model_gcn.parameters(), lr=args.lr, weight_decay=1e-5)
    optimizer_ae = optim.Adam(model_ae_image.parameters(), lr=args.lr, weight_decay=1e-5)

    # retrieval_targets_onehot = retrieval_dataloader.dataset.get_onehot_targets().to(args.device)
    retrieval_targets_onehot = retrieval_dataloader.dataset.get_onehot_targets()

    # B_all = torch.randn(num_retrieval, code_length).to(args.device)
    B_all = torch.zeros(num_retrieval, code_length).to(args.device)

    # U_image = torch.zeros(args.num_samples, code_length).to(args.device)
    # U_label = torch.zeros(args.num_samples, code_length).to(args.device)
    # U_gcn = torch.zeros(args.num_samples, code_length).to(args.device)

    mAP_best = 0
    mAP_best_res = None

    start = time.time()
    for it in range(args.max_iter):
        U_image_sample = torch.zeros(args.num_samples, code_length).to(args.device)
        U_label_sample = torch.zeros(args.num_samples, code_length).to(args.device)
        U_latent_sample = torch.zeros(args.num_samples, code_length).to(args.device)

        iter_start = time.time()
        # Sample training data for cnn learning
        train_dataloader, sample_index_in_all = sample_dataloader(retrieval_dataloader, args.num_samples, args.batch_size, args.root, args.dataset)
        sample_index_in_all = sample_index_in_all.to(args.device)
        # Create Similarity matrix
        # train_targets_onehot = train_dataloader.dataset.get_onehot_targets().to(args.device)
        train_targets_onehot = train_dataloader.dataset.get_onehot_targets()

        S_ = (train_targets_onehot @ retrieval_targets_onehot.t() > 0).float()
        S_neg1 = torch.where(S_ == 1, torch.full_like(S_, 1), torch.full_like(S_, -1))

        # Soft similarity matrix, benefit to converge
        r = S_neg1.sum() / (1 - S_neg1).sum()
        S_neg1 = S_neg1 * (1 + r) - r

        S_neg1 = S_neg1.to(args.device)
        train_targets_onehot = train_targets_onehot.to(args.device)

        for epoch in tqdm(range(args.max_epoch)):
            for batch, (data, targets, batch_index_in_sample) in enumerate(train_dataloader):
                data, targets, batch_index_in_sample = data.to(args.device), targets.to(args.device), batch_index_in_sample.to(args.device)

                batch_sim = ((train_targets_onehot[batch_index_in_sample] @
                              train_targets_onehot[batch_index_in_sample].T) > 0).float()
                batch_sim_neg1 = torch.where(batch_sim == 1, torch.full_like(batch_sim, 1), torch.full_like(batch_sim, -1))

                optimizer_image.zero_grad()
                output = model_image(data)
                U_image = output

                optimizer_label.zero_grad()
                output = model_label(train_targets_onehot[batch_index_in_sample])
                U_label = output

                optimizer_ae.zero_grad()

                output_ae_image, latent_space = model_ae_image(U_label.detach())

                optimizer_gcn.zero_grad()
                output = model_gcn(latent_space, batch_sim)
                U_latent = output

                loss = criterion_all(U_latent, U_image, U_label,
                                     output_ae_image, U_image.detach(),
                                     B_all[sample_index_in_all[batch_index_in_sample], :],
                                     batch_sim_neg1)
                loss.backward()
                optimizer_gcn.step()
                optimizer_image.step()
                optimizer_label.step()
                optimizer_ae.step()

                U_latent_sample[batch_index_in_sample, :] = U_latent.detach()
                U_image_sample[batch_index_in_sample, :], U_label_sample[batch_index_in_sample, :] = U_image.detach(), U_label.detach()


        def calc_dcc():
            B, U, S = B_all[sample_index_in_all, :], U_image_sample, S_neg1[:, sample_index_in_all]

            Q = (code_length * S).t() @ U + args.gamma * U

            for bit in range(code_length):
                q = Q[:, bit]
                u = U[:, bit]
                B_prime = torch.cat((B[:, :bit], B[:, bit + 1:]), dim=1)
                U_prime = torch.cat((U[:, :bit], U[:, bit + 1:]), dim=1)

                B[:, bit] = (q.t() - B_prime @ U_prime.t() @ u.t() + args.alpha * U_label_sample[:, bit] + args.beta*U_latent_sample[:, bit]).sign()
            return B

        B_all[sample_index_in_all, :] = calc_dcc()

        logger.debug('[iter:{}/{}][iter_time:{:.2f}]'.format(it + 1, args.max_iter, time.time() - iter_start))

        if (it + 1) % args.eval_iter == 0:
            # Evaluate
            query_code = generate_code(model_image, query_dataloader, code_length, args.device)
            mAP = evaluate.mean_average_precision(
                query_code.to(args.device),
                B_all,
                query_dataloader.dataset.get_onehot_targets().to(args.device),
                retrieval_targets_onehot.to(args.device),
                args.device,
                args.topk,
            )
            logger.info("[Evaluation][dataset:{}][bits:{}][iter:{}/{}][mAP:{:.4f}]".format(args.dataset, code_length, it + 1, args.max_iter, mAP))
            if mAP > mAP_best:
                mAP_best = mAP

                mAP_best_res = [query_code.cpu(), B_all.cpu(), query_dataloader.dataset.get_onehot_targets().cpu(), retrieval_targets_onehot.cpu()]

    logger.info('[Training time:{:.2f}]'.format(time.time() - start))

    if mAP_best_res is not None:
        query_code, database_code, query_targets, database_targets = mAP_best_res

        # Save checkpoints
        gen_name = lambda name: os.path.join(logdir, f'{args.dataset}-{code_length}bits-{mAP_best}-{name}.t')
        torch.save(query_code, gen_name("query_code"))
        torch.save(database_code, gen_name("database_code"))
        torch.save(query_targets, gen_name("query_targets"))
        torch.save(database_targets, gen_name("database_targets"))
        torch.save(model_image, gen_name("model_image"))

    return mAP_best


def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code

    Args
        dataloader(torch.utils.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): Using gpu or cpu.

    Returns
        code(torch.Tensor): Hash code.
    """
    model.eval()
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        for data, _, index in dataloader:
            data = data.to(device)
            hash_code = model(data)
            code[index, :] = hash_code.sign().cpu()

    model.train()
    return code
