from nets import DiffusionScore, SimpleMLP, ModelEnsemble
from lib.sdes import VariancePreservingSDE
from utils import *
from likelihood import *
import argparse
import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from math_benchmark import *
import torch.optim as optim
from torch.distributions.normal import Normal
import time
import higher

@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)
with suppress_output():
    import design_bench


def classifier_training(args):
    #data
    if args.task in ['Ackley', 'Rastrigin', 'Rosenbrock']:
        task = MathBench(args.task)
    else:
        task = design_bench.make(args.task)
    task_y0 = task.y
    task_x, task_y, shape0 = process_data(task, args.task, task_y0)
    task_x = torch.Tensor(task_x).to(args.device)
    task_y = torch.Tensor(task_y).to(args.device)
    L = task_x.shape[0]
    indexs = torch.randperm(L)
    task_x = task_x[indexs]
    task_y = task_y[indexs]
    train_L = int(L * 0.90)
    # normalize labels
    train_labels0 = task_y[0: train_L]
    valid_labels = task_y[train_L:]
    # load logits
    train_logits0 = task_x[0: train_L]
    valid_logits = task_x[train_L:]
    T = int(train_L / args.bs) + 1
    # define model
    classifier = SimpleMLP(task_x.shape[1]).to(args.device)
    opt = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    # begin training
    print('begin training')
    best_pcc = -torch.Tensor([float('inf')]).to(args.device)
    best_epoch = -1
    for e in range(args.epochs):
        print('epoch', e)
        # adjust lr
        adjust_learning_rate(opt, args.lr, e, args.epochs)
        # random shuffle
        indexs = torch.randperm(train_L)
        train_logits = train_logits0[indexs]
        train_labels = train_labels0[indexs]
        tmp_loss = 0
        for t in range(T):
            x_batch = train_logits[t * args.bs:(t + 1) * args.bs, :]
            y_batch = train_labels[t * args.bs:(t + 1) * args.bs]
            pred_mu, pred_std = classifier(x_batch)
            #print("mean", pred_mu, "std", pred_std)
            dis = Normal(pred_mu.squeeze(), pred_std.squeeze())
            #torch.Size([128, 1]) torch.Size([128, 1])
            #loss = torch.mean(torch.pow(pred - y_batch, 2))
            loss = torch.mean(-dis.log_prob(y_batch.squeeze()))
            tmp_loss = tmp_loss + loss.data
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            valid_preds, _ = classifier(valid_logits)
            valid_pcc = compute_pcc(valid_preds.squeeze(), valid_labels.squeeze())
        print("epoch {} training loss {} pcc  {} best pcc {} from best epoch {}".format(e, tmp_loss / T, valid_pcc, best_pcc, best_epoch))
        if valid_pcc > best_pcc:
            best_pcc = valid_pcc
            print("epoch {} has the best pcc {}".format(e, best_pcc))
            best_epoch = e
            classifier = classifier.to(torch.device('cpu'))
            torch.save(classifier.state_dict(), "model/" + args.task + "_proxy_" + str(args.seed) + ".pt")
            classifier = classifier.to(args.device)
    print(args.task)
    print('classifier training complete')

def diffusion_training_elbo(args):
    #data
    if args.task in ['Ackley', 'Rastrigin', 'Rosenbrock']:
        task = MathBench(args.task)
    else:
        task = design_bench.make(args.task)
    task_y0 = task.y
    task_x, task_y, shape0 = process_data(task, args.task, task_y0)
    task_x = torch.Tensor(task_x).to(args.device)
    task_y = torch.Tensor(task_y).to(args.device)
    L = task_x.shape[0]
    indexs = torch.randperm(L)
    task_x = task_x[indexs]
    task_y = task_y[indexs]
    train_L = int(L * 0.90)
    # normalize labels
    train_labels0 = task_y[0: train_L]
    train_weights0 = get_weights(train_labels0.cpu().numpy())
    train_weights0 = torch.Tensor(train_weights0).to(args.device)
    valid_labels = task_y[train_L:]
    # load logits
    train_logits0 = task_x[0: train_L]
    valid_logits = task_x[train_L:]
    T = int(train_L / args.bs) + 1
    # define model
    model = DiffusionScore(task_x=task_x,
                           task_y=task_y).to(args.device)
    opt = torch.optim.Adam(model.gen_sde.parameters(), lr=args.lr)
    # begin training
    print('begin training')
    best_elbo = torch.tensor(-float('inf'))
    best_epoch = 0
    for e in range(args.epochs):
        print('epoch', e)
        adjust_learning_rate(opt, args.lr, e, args.epochs)
        # random shuffle
        indexs = torch.randperm(train_L)
        train_logits = train_logits0[indexs]
        train_labels = train_labels0[indexs]
        train_weights = train_weights0[indexs]
        tmp_loss = 0
        for t in range(T):
            x_batch = train_logits[t * args.bs:(t + 1) * args.bs, :]
            y_batch = train_labels[t * args.bs:(t + 1) * args.bs]
            w_batch = train_weights[t * args.bs:(t + 1) * args.bs]
            #(128, d), (128, 1), (128, 1)
            loss = model.training_step([x_batch, y_batch, w_batch])
            tmp_loss = tmp_loss + loss.data
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            elbo = torch.mean(model.validation_step_elbo([valid_logits, valid_labels]))
        print(args.task)
        print("epoch {} training elbo {} valid eblo {} best valid elbo {} from best epoch {}".format(e, tmp_loss / T, elbo, best_elbo, best_epoch))
        if elbo > best_elbo:
            best_elbo = elbo
            best_epoch = e
            print("epoch {} has the best elbo {}".format(e, best_elbo))
            model.gen_sde.a = model.gen_sde.a.to(torch.device('cpu'))
            torch.save(model.gen_sde.a.state_dict(), "model/" + args.task + "_score_estimator_elbo_" + str(args.seed) + ".pt")
            model.gen_sde.a = model.gen_sde.a.to(args.device)
    print('training complete')
def classifier_finetuning(args):
    # data
    if args.task in ['Ackley', 'Rastrigin', 'Rosenbrock']:
        task = MathBench(args.task)
    else:
        task = design_bench.make(args.task)
    task_y0 = task.y
    task_x, task_y, shape0 = process_data(task, args.task, task_y0)
    kde = gaussian_kde(task_y.squeeze())
    device = args.device
    task_x = torch.Tensor(task_x).to(args.device)
    task_y = torch.Tensor(task_y).to(args.device)

    indexs = torch.argsort(task_y.squeeze())
    index = indexs[-128:]
    valid_logits = task_x[index]
    #128xd

    #define model
    classifier = SimpleMLP(task_x.shape[1]).to(args.device)
    state_dict = torch.load("model/" + args.task + "_proxy_" + str(args.seed) + ".pt")
    classifier.load_state_dict(state_dict)
    #load diffusion model
    model = DiffusionScore(task_x=task_x,
                           task_y=task_y).to(args.device)
    state_dict = torch.load("model/" + args.task + "_score_estimator_elbo_" + str(1) + ".pt")
    model.gen_sde.a.load_state_dict(state_dict)
    #generate samples
    sample_n=300
    def identify_candidates(sample_n=300):
        candidate = copy.deepcopy(valid_logits.data)
        candidate.requires_grad = True
        grad_candidate = torch.zeros(sample_n * candidate.shape[0], candidate.shape[1]).to(device)
        if args.task in ['TFBind8-Exact-v0', 'TFBind10-Exact-v0', 'CIFARNAS-Exact-v0']:
            candidate_opt = optim.Adam([candidate], lr=0.1)
        else:
            candidate_opt = optim.Adam([candidate], lr=0.001)
        for i in range(1, sample_n + 1):
            loss = torch.sum(-classifier(candidate)[0])
            candidate_opt.zero_grad()
            loss.backward()
            candidate_opt.step()
            grad_candidate[(i-1)*candidate.shape[0]: i*candidate.shape[0], :] = candidate.data
        return grad_candidate[(sample_n-1)*candidate.shape[0]: sample_n*candidate.shape[0], :]
    def identify_gendata(candidates):
        my_likelihood_fn = get_likelihood_fn(model.inf_sde)
        sample_x_n = 50
        diffusion_n = 20
        # [128*sample_x_n, D+2]
        gendata = torch.zeros(sample_x_n * candidates.shape[0], candidates.shape[1] + 2).to(device)
        for x_start in range(candidates.shape[0]):
            start_time = time.time()
            print('processing', x_start)
            input_x = candidates[x_start:x_start + 1].repeat(sample_x_n, 1)
            #print('shape match1', gendata[sample_x_n * x_start: sample_x_n * (x_start + 1), 0:-2].shape, input_x.shape)
            gendata[sample_x_n*x_start: sample_x_n*(x_start + 1), 0:-2] = input_x.data

            pred_mu, pred_std = classifier(input_x)
            dis = Normal(pred_mu.squeeze(), pred_std.squeeze())
            proxy_y_x_sample = dis.sample()
            #print('shape match2', gendata[sample_x_n * x_start: sample_x_n * (x_start + 1), -2].shape, proxy_y_x_sample.shape)
            gendata[sample_x_n * x_start: sample_x_n * (x_start + 1), -2] = proxy_y_x_sample.data

            diffusion_y_p = kde.evaluate(proxy_y_x_sample.cpu().numpy())
            diffusion_y_logp = torch.log(torch.from_numpy(diffusion_y_p) + 1e-9).to(device)

            diffusion_x_y_logp = 0
            diffusion_x_logp = 0
            for i in range(diffusion_n):
                diffusion_x_y_logp_i = my_likelihood_fn(model, input_x, proxy_y_x_sample)
                diffusion_x_y_logp = diffusion_x_y_logp + diffusion_x_y_logp_i
                diffusion_x_logp_i = my_likelihood_fn(model, input_x, torch.zeros_like(proxy_y_x_sample))
                diffusion_x_logp = diffusion_x_logp + diffusion_x_logp_i
            diffusion_x_y_logp = diffusion_x_y_logp / diffusion_n
            diffusion_x_logp = diffusion_x_logp/diffusion_n

            diffusion_y_x_logp = diffusion_x_y_logp + diffusion_y_logp - diffusion_x_logp
            print("diffusion_y_x_logp", diffusion_y_x_logp)
            #diffusion_y_x_logp = torch.log(mysoftmax(diffusion_y_x_logp) + 1e-9)
            #print('shape match3', gendata[sample_x_n * x_start: sample_x_n * (x_start + 1), -1].shape, diffusion_y_x_logp.shape)
            gendata[sample_x_n * x_start: sample_x_n * (x_start + 1), -1] = diffusion_y_x_logp.data
            print("time cost", time.time() - start_time)
            #[xs, ys, diffusion_y_x_logp]
        torch.save(gendata, "model/" + args.task + "_gendata.pt")
    if os.path.isfile("model/" + args.task + "_gendata.pt"):
        print('loading data')
        gendata = torch.load("model/" + args.task + "_gendata.pt", map_location=device)
    else:
        print('generating data')
        candidates = identify_candidates(sample_n)
        print('identify_candidates', candidates.shape)
        #[128, D]
        identify_gendata(candidates)
        gendata = torch.load("model/" + args.task + "_gendata.pt", map_location=device)
    print('gendata shape', gendata.shape)
    # if 1:
    #     return 1
    #128xd
    def get_kl(classifier, gendata):
        sample_x_n = 50
        weighted_loss = 0
        L = int(gendata.shape[0]/sample_x_n)
        for x_start in range(L):
            input_x = gendata[sample_x_n*x_start: sample_x_n*(x_start + 1), 0:-2]
            pred_mu, pred_std = classifier(input_x)
            # print('pred shape', pred_mu.shape, pred_std.shape)
            dis = Normal(pred_mu.squeeze(), pred_std.squeeze())
            #proxy_y_x_sample = dis.sample()
            proxy_y_x_sample = gendata[sample_x_n * x_start: sample_x_n * (x_start + 1), -2]
            # print('proxy_y_x_sample shape', proxy_y_x_sample.shape)
            proxy_y_x_logp = dis.log_prob(proxy_y_x_sample).to(torch.float64)
            #print("proxy_y_x_logp", proxy_y_x_logp)
            #proxy_y_x_logp = torch.log(mysoftmax(proxy_y_x_logp) + 1e-9)
            diffusion_y_x_logp = gendata[sample_x_n * x_start: sample_x_n * (x_start + 1), -1]

            #print("proxy_y_x_logp", proxy_y_x_logp, "diffusion_y_x_logp", diffusion_y_x_logp)
            weight = (1 + proxy_y_x_logp - diffusion_y_x_logp).data
            #print("weight", weight)
            #weight = torch.clip(weight, min=1-2.3026, max=1+2.3026)#10
            weight = torch.clip(weight, min=1 - 4.6052, max=1 + 4.6052)# 100
            #weight = torch.clip(weight, min=1 - 6.9078, max=1 + 6.9078)# 1000
            #weight = torch.clip(weight, min=1-0.4055, max=1+0.4055)
            #weight = torch.clip(weight, min=1.0, max=1.0)
            #weight = torch.clip(weight, min=1-0.6931, max=1+0.6931)
            #print(weight)
            weighted_loss = weighted_loss + torch.dot(weight, proxy_y_x_logp) / diffusion_y_x_logp.shape[0]
            #print('weight', torch.mean(weight), torch.std(weight), torch.min(weight), torch.max(weight))
        return weighted_loss/L

    def fine_tune(classifier, gendata):
        nonlocal task_x
        nonlocal task_y

        L = task_x.shape[0]
        indexs = torch.randperm(L)
        task_x = task_x[indexs]
        task_y = task_y[indexs]
        train_L = int(L * 0.90)
        # normalize labels
        train_labels0 = task_y[0: train_L]
        valid_labels = task_y[train_L:]
        # load logits
        train_logits0 = task_x[0: train_L]
        valid_logits = task_x[train_L:]
        args.bs = 256
        T = int(train_L / args.bs) + 1

        opt = torch.optim.Adam(classifier.parameters(), lr=args.lr)
        # begin training
        print('begin fine-tuning')
        best_pcc = -torch.Tensor([float('inf')]).to(args.device)
        best_epoch = -1

        alpha = torch.Tensor([0.001]).to(device)
        alpha.requires_grad = True
        opt_alpha = torch.optim.Adam([alpha], lr=0.001)

        args.epochs = 20
        with torch.no_grad():
            valid_preds, _ = classifier(valid_logits)
            valid_pcc = compute_pcc(valid_preds.squeeze(), valid_labels.squeeze())
            print('initial pcc', valid_pcc)
        for e in range(args.epochs):
            print('epoch', e)
            # adjust lr
            adjust_learning_rate(opt, args.lr, e, args.epochs)
            # random shuffle
            indexs = torch.randperm(train_L)
            train_logits = train_logits0[indexs]
            train_labels = train_labels0[indexs]
            tmp_loss = 0
            for t in range(T):
                x_batch = train_logits[t * args.bs:(t + 1) * args.bs, :]
                y_batch = train_labels[t * args.bs:(t + 1) * args.bs]

                #if False:
                with higher.innerloop_ctx(classifier, opt) as (fmodel, diffopt):
                    pred_mu, pred_std = fmodel(x_batch)
                    dis = Normal(pred_mu.squeeze(), pred_std.squeeze())

                    loss_t = torch.mean(-dis.log_prob(y_batch.squeeze())) + alpha * get_kl(fmodel, gendata)
                    diffopt.step(loss_t)

                    pred_mu_v, pred_std_v = fmodel(valid_logits)
                    dis_v = Normal(pred_mu_v.squeeze(), pred_std_v.squeeze())
                    loss_v = torch.mean(-dis_v.log_prob(valid_labels.squeeze()))
                    opt_alpha.zero_grad()
                    loss_v.backward()
                    opt_alpha.step()
                    alpha.data = torch.clip(alpha.data, min=0.0, max=0.01)
                    #alpha.data = torch.clip(alpha.data, min=0.0, max=0.10)
                    #grad = torch.autograd.grad(loss_v, alpha)[0].data
                    #alpha = torch.clip(alpha - 0.1 * grad, min=0, max=2.0)
                    print('alpha', alpha)

                pred_mu, pred_std = classifier(x_batch)
                dis = Normal(pred_mu.squeeze(), pred_std.squeeze())

                loss = torch.mean(-dis.log_prob(y_batch.squeeze())) + alpha.data * get_kl(classifier, gendata)
                #loss = torch.mean(-dis.log_prob(y_batch.squeeze())) + 0.001 * get_kl(classifier, gendata)
                tmp_loss = tmp_loss + loss.data
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                valid_preds, _ = classifier(valid_logits)
                valid_pcc = compute_pcc(valid_preds.squeeze(), valid_labels.squeeze())
                #exit(0)
            print("epoch {} training loss {} pcc  {} best pcc {} from best epoch {}".format(e, tmp_loss / T, valid_pcc,
                                                                                            best_pcc, best_epoch))
            if valid_pcc > best_pcc:
                best_pcc = valid_pcc
                print("epoch {} has the best pcc {}".format(e, best_pcc))
                best_epoch = e
        classifier = classifier.to(torch.device('cpu'))
        torch.save(classifier.state_dict(), "model/" + args.task + "_fine_tuned_proxy_" + str(args.seed) + ".pt")
        print(args.task)
        print('classifier fine-tuning complete')

    fine_tune(classifier, gendata)

def both(args=None):
    # data
    if args.task in ['Ackley', 'Rastrigin', 'Rosenbrock']:
        task = MathBench(args.task)
    else:
        task = design_bench.make(args.task)
    task_y0 = task.y
    task_x, task_y, shape0 = process_data(task, args.task, task_y0)
    task_x = torch.Tensor(task_x).to(args.device)
    task_y = torch.Tensor(task_y).to(args.device)
    #
    classifier = SimpleMLP(task_x.shape[1]).to(args.device)
    if args.fine_tuned:
        if args.proxy == "ours":
            state_dict = torch.load("model/" + args.task + "_fine_tuned_proxy_" + str(args.seed) + ".pt")
        elif args.proxy == "coms":
            state_dict = torch.load("model/" + args.task + "_proxy_coms_" + str(1) + ".pt")
        elif args.proxy == "roma":
            state_dict = torch.load("model/" + args.task + "_proxy_roma_" + str(1) + ".pt")
    else:
        state_dict = torch.load("model/" + args.task + "_proxy_" + str(args.seed) + ".pt")
    classifier.load_state_dict(state_dict)

    #
    classifier_s = SimpleMLP(task_x.shape[1]).to(args.device)
    state_dict = torch.load("model/" + args.task + "_proxy_" + str(args.seed) + ".pt")
    classifier_s.load_state_dict(state_dict)

    # define model
    model = DiffusionScore(task_x=task_x,
                           task_y=task_y).to(args.device)
    state_dict = torch.load("model/" + args.task + "_score_estimator_elbo_" + str(1) + ".pt")
    model.gen_sde.a.load_state_dict(state_dict)

    results = np.load("Figure_New/hyper/hyper.npy", allow_pickle=True).item()

    learn_max_scores = []
    learn_med_scores = []

    for test_n in range(50):
        print("test_n", test_n)
        num_candidates = 128
        x_0 = torch.randn(num_candidates, task_x.shape[-1], device=args.device)
        y_ = torch.ones(num_candidates).to(args.device) * torch.max(task_y) * args.y_ratio

        x_s = heun_sampler_ode_hyper(model, x_0, ya=y_, num_steps=args.num_steps, classifier=classifier,
                               classifier_s=classifier_s, gamma_learn=True, sample_lr=args.sample_lr)
        max_score, med_score = evaluate_sample(task, x_s, args.task, shape0)
        print('ode learn score', max_score, med_score)
        learn_max_scores.append(max_score)
        learn_med_scores.append(med_score)

    learn_max_scores = np.array(learn_max_scores)
    learn_med_scores = np.array(learn_med_scores)
    key = args.task + "_" + str(args.y_ratio) + "_" + str(args.num_steps) + "_" + str(args.sample_lr)
    results[key] = [learn_max_scores, learn_med_scores]
    np.save("Figure_New/hyper/hyper.npy", results)

    print("learn_scores", np.mean(learn_max_scores), np.std(learn_max_scores), np.mean(learn_med_scores), np.std(learn_med_scores))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust Guided Diffusion")
    parser.add_argument('--mode', choices=['classifier_training', 'classifier_finetuning',
                                           'diffusion_training_elbo', 'both'],
                        type=str, default='both')
    parser.add_argument('--proxy', choices=['ours', 'roma', 'coms'], type=str, default='ours')
    parser.add_argument('--task', choices=['Superconductor-RandomForest-v0', 'TFBind8-Exact-v0', 'HopperController-Exact-v0',
                                           'AntMorphology-Exact-v0', 'DKittyMorphology-Exact-v0', 'AntMorphology-Exact-v0',
                                           'TFBind10-Exact-v0', 'CIFARNAS-Exact-v0', 'Rosenbrock'],
                        type=str, default='AntMorphology-Exact-v0')
    parser.add_argument('--bs', default=256, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--device', default="", type=str)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--ratio', default=0.5, type=float)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--vtype', default="rademacher", type=str)
    parser.add_argument('--gamma_learn', action='store_true')
    parser.add_argument('--fine_tuned', action='store_true')

    #hyper parameter
    parser.add_argument('--y_ratio', choices=[0.5, 1.0, 1.5, 2.0, 2.5], default=1.0, type=float)
    parser.add_argument('--num_steps', choices=[500, 750, 1000, 1250, 1500], default=1000, type=int)
    parser.add_argument('--sample_lr', choices=[2.5e-3, 5.0e-3, 7.5e-3, 1.0e-2, 1.25e-2, 1.5e-2, 2.0e-2, 4.0e-2], default=1e-2, type=float)


    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)
    load_y(args.task)
    print(args)
    if args.mode == 'classifier_training':
        classifier_training(args)
    elif args.mode == 'diffusion_training_elbo':
        diffusion_training_elbo(args)
    elif args.mode == 'classifier_finetuning':
        classifier_finetuning(args)
    elif args.mode == 'both':
        both(args)
