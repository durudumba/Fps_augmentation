import torch
from tqdm import tqdm
from matplotlib import pyplot as plt

def train(args, model, train_loader, test_loader):
    
    ## Optimizer 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    ## 학습하기
    count = 0
    train_loss_plot, eval_loss_plot = [], []

    for epoch in tqdm(range(args.epochs), desc="Epoch"):
        
        # Train

        train_loss = 0
        optimizer.zero_grad()

        model.train()
        train_iterator = tqdm(enumerate(train_loader), total=len(train_loader), desc="Train")

        for i, batch_item in train_iterator:
            
            future_data, past_data = batch_item

            past_data = past_data.float().to(args.device)
            future_data = future_data.float().to(args.device)

            loss = model(past_data, future_data)
            train_loss += loss.mean().item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_iterator.set_postfix({
                "train_loss" : float(loss),
                "train_mean_loss" : train_loss/(i+1)
            })
        train_loss_plot.append(train_loss / len(train_loader))


        # Validation

        eval_loss = 0
        
        model.eval()
        val_iterator = tqdm(enumerate(test_loader), total=len(test_loader), desc="Validation")
        with torch.no_grad():
            for i, batch_item in val_iterator:

                future_data, past_data = batch_item
                
                past_data = past_data.float().to(args.device)
                future_data = future_data.float().to(args.device)

                loss = model(past_data, future_data)
                eval_loss += loss.mean().item()

                val_iterator.set_postfix({
                    "eval_loss" : float(loss),
                    "eval_mean_loss" : eval_loss/(i+1)
                })
        eval_loss_plot.append(eval_loss / len(test_loader))

    plt.figure(figsize=(20,7))
    plt.title("loss")
    plt.plot(range(args.epochs), train_loss_plot, eval_loss_plot)
    plt.legend(['train', 'eval'])

    plt.savefig('./result/loss_graph.png')
    return model
