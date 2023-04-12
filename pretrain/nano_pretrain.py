import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import set_seed
from mingpt.model import GPT
from mingpt.trainer import Trainer
from dataset.sort_dataset import SortDataset


def batch_end_callback(obj):
    if obj.iter_num % 100 == 0:
        print(f"iter_dt {obj.iter_dt * 1000:.2f}ms; iter {obj.iter_num}: train loss {obj.loss.item():.5f}")


def eval_split(obj, split, max_batches, train_set, test_set):
    dataset = {'train': train_set, 'test': test_set}[split]
    n = train_dataset.length # naugy direct access shrug
    results = []
    mistakes_printed_already = 0
    loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)
    for b, (x, y) in enumerate(loader):
        x = x.to(obj.device)
        y = y.to(obj.device)
        # isolate the input pattern alone
        inp = x[:, :n]
        sol = y[:, -n:]
        # let the model sample the rest of the sequence
        cat = model.generate(inp, n, do_sample=False) # using greedy argmax, not sampling
        sol_candidate = cat[:, n:] # isolate the filled in sequence
        # compare the predicted sequence to the true sequence
        correct = (sol == sol_candidate).all(1).cpu() # Software 1.0 vs. Software 2.0 fight RIGHT on this line haha
        for i in range(x.size(0)):
            results.append(int(correct[i]))
            if not correct[i] and mistakes_printed_already < 3: # only print up to 5 mistakes to get a sense
                mistakes_printed_already += 1
                print("GPT claims that %s sorted is %s but gt is %s" % (inp[i].tolist(), sol_candidate[i].tolist(),
                                                                        sol[i].tolist()))
        if max_batches is not None and b+1 >= max_batches:
            break
    rt = torch.tensor(results, dtype=torch.float)
    print("%s final score: %d/%d = %.2f%% correct" % (split, rt.sum(), len(results), 100*rt.mean()))
    return rt.sum()


def verify(obj, train_set, test_set):
    # run a lot of examples from both train and test through the model and verify the output correctness
    with torch.no_grad():
        train_score = eval_split(obj, 'train', max_batches=50, train_set=train_set, test_set=test_set)
        test_score = eval_split(obj, 'test', max_batches=50, train_set=train_set, test_set=test_set)

    # let's run a random given sequence through the model as well
    n = train_set.length  # naugy direct access shrug
    inp = torch.tensor([[0, 0, 2, 1, 0, 1]], dtype=torch.long).to(obj.device)
    assert inp[0].nelement() == n
    with torch.no_grad():
        cat = model.generate(inp, n, do_sample=False)
    sol = torch.sort(inp[0])[0]
    sol_candidate = cat[:, n:]
    print('input sequence  :', inp.tolist())
    print('predicted sorted:', sol_candidate.tolist())
    print('gt sort         :', sol.tolist())
    print('matches         :', bool((sol == sol_candidate).all()))


if __name__ == "__main__":
    set_seed(3407)

    train_dataset = SortDataset('train')
    test_dataset = SortDataset('test')

    # create a GPT instance
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-nano'
    model_config.vocab_size = train_dataset.get_vocab_size()
    model_config.block_size = train_dataset.get_block_size()
    model = GPT(model_config)

    # create a Trainer object
    train_config = Trainer.get_default_config()
    train_config.learning_rate = 5e-4  # the model we're using is so small that we can go a bit faster
    train_config.max_iters = 2000
    train_config.num_workers = 0
#    train_config.device = 'cuda'
    trainer = Trainer(train_config, model, train_dataset)

    trainer.set_callback('on_batch_end', batch_end_callback)

    trainer.run()

    verify(trainer, train_dataset, test_dataset)

