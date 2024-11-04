import torch

from src.defenses.competitors.a_vae.model import StyledGenerator

if __name__ == '__main__':

    generator_dir = '/media/dserez/runs/adversarial/competitors/a_vae/celeba_identities/checkpoint/000002.pt'

    g_running = StyledGenerator(output_size=64).cuda()
    g_running.eval()
    g_checkpoint = torch.load(generator_dir)
    g_running.load_state_dict(g_checkpoint)

    t = 0.41
    iteration = 0
    r1, r2 = 0, 0
    j = 0
    m = 0
    epsilon = 0
    alpha = (4 / 255.) / 0.5

    batch_size = 10

    pairs = open(pairs_r, 'r')
    lines = pairs.readlines()[1:]
    cls = 'lfw_clean'
    inputs = torch.zeros([batch_size, 3, 224, 224], dtype=torch.float32).cuda()
    targets = torch.zeros([batch_size, 3, 224, 224], dtype=torch.float32).cuda()

    # c&w attack
    cw_t = cw.L2Adversary(targeted=True,
                          confidence=0.0,
                          search_steps=10,
                          optimizer_lr=10)

    for i in range(300):
        line = lines[n * 600 + i + 300]
        name1, n1, name2, n2 = line.split('\t')
        n1, n2 = int(n1), int(n2)
        input = get_image(os.path.join(dataset_dir, name1), '%s_%04d.jpg' % (name1, n1), transform)
        target = get_image(os.path.join(dataset_dir, name2), '%s_%04d.jpg' % (name2, n2), transform)
        inputs[j] = input
        targets[j] = target
        j += 1
        if j % batch_size == 0:
            j = 0
            inputs = fgsm_w(net, g_running, inputs, targets, epsilon, alpha, iteration, t=False)

            if avae_defense:
                inputs = recon(128, g_running, inputs.cuda())
            fea1 = net(inputs)
            fea2 = net(targets)
            cosS = CosSimi(fea1, fea2).data.cpu().numpy()
            r2 += np.sum(cosS < t)

    p1 = round(float(r1) / (300 * (n + 1)), 4)
    p2 = round(float(r2) / (300 * (n + 1)), 4)
