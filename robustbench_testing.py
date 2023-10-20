from robustbench.data import load_cifar10
from robustbench.utils import load_model, clean_accuracy
from autoattack import AutoAttack


# https://github.com/RobustBench/robustbench
# https://robustbench.github.io/
def main():

    base_path = '/media/dserez/code/adversarial/'

    # avail datasets: CIFAR10 - CIFAR100 - IMAGENET??
    dataset = 'cifar10'
    x_test, y_test = load_cifar10(n_examples=100, data_dir=base_path)

    # how to load a model for robust defense
    model_name = 'Peng2023Robust'  # check leaderboard (FirstAuthorYearFirstWord)
    threat_model = 'Linf'  # L2 Linf
    defense_model = load_model(model_name=model_name, model_dir=base_path, dataset=dataset, threat_model=threat_model)

    clean_acc = clean_accuracy(defense_model, x_test, y_test)
    print(f'model: {model_name} has clean accuracy on {dataset} of {clean_acc}')

    # attacking a model with AutoAttack
    adversary = AutoAttack(defense_model, norm='Linf', eps=8 / 255, device='cpu',
                           version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])

    adversary.apgd.n_restarts = 1
    x_adv = adversary.run_standard_evaluation(x_test, y_test)

    print(x_adv)


if __name__ == '__main__':
    main()
