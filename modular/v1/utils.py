import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, model, loss_fn, optimizer, train_loader, val_loader, scheduler=None, 
                 callbacks=None, device=None):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        # scheduler 추가
        self.scheduler = scheduler
        self.device = device
        # 현재 learning rate 변수 추가
        self.current_lr = self.optimizer.param_groups[0]['lr']
        #checkpoint와 early stopping 클래스들을 list로 받음. 
        self.callbacks = callbacks
        
    def train_epoch(self, epoch):
        self.model.train()

        # running 평균 loss 계산.
        accu_loss = 0.0
        running_avg_loss = 0.0
        # 정확도, 정확도 계산을 위한 전체 건수 및 누적 정확건수
        num_total = 0.0
        accu_num_correct = 0.0
        accuracy = 0.0
        # tqdm으로 실시간 training loop 진행 상황 시각화
        with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch+1} [Training..]", leave=True) as progress_bar:
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                # 반드시 to(self.device). to(device) 아님.
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # batch 반복 시 마다 누적  loss를 구하고 이를 batch 횟수로 나눠서 running 평균 loss 구함.
                accu_loss += loss.item()
                running_avg_loss = accu_loss /(batch_idx + 1)

                # accuracy metric 계산
                # outputs 출력 예측 class값과 targets값 일치 건수 구하고
                num_correct = (outputs.argmax(-1) == targets).sum().item()
                # 배치별 누적 전체 건수와 누적 전체 num_correct 건수로 accuracy 계산  
                num_total += inputs.shape[0]
                accu_num_correct += num_correct
                accuracy = accu_num_correct / num_total

                #tqdm progress_bar에 진행 상황 및 running 평균 loss와 정확도 표시
                progress_bar.update(1)
                if batch_idx % 20 == 0 or (batch_idx + 1) == progress_bar.total:  # 20 batch 횟수마다 또는 맨 마지막 batch에서 update
                    progress_bar.set_postfix({"Loss": running_avg_loss,
                                              "Accuracy": accuracy})

        if (self.scheduler is not None) and (not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)):
            self.scheduler.step()
            self.current_lr = self.scheduler.get_last_lr()[0]
        
        return running_avg_loss, accuracy

    def validate_epoch(self, epoch):
        if not self.val_loader:
            return None

        self.model.eval()

        # running 평균 loss 계산.
        accu_loss = 0
        running_avg_loss = 0
        # 정확도, 정확도 계산을 위한 전체 건수 및 누적 정확건수
        num_total = 0.0
        accu_num_correct = 0.0
        accuracy = 0.0
        current_lr = self.optimizer.param_groups[0]['lr']
        with tqdm(total=len(self.val_loader), desc=f"Epoch {epoch+1} [Validating]", leave=True) as progress_bar:
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    outputs = self.model(inputs)

                    loss = self.loss_fn(outputs, targets)
                    # batch 반복 시 마다 누적  loss를 구하고 이를 batch 횟수로 나눠서 running 평균 loss 구함.
                    accu_loss += loss.item()
                    running_avg_loss = accu_loss /(batch_idx + 1)

                    # accuracy metric 계산
                    # outputs 출력 예측 class값과 targets값 일치 건수 구하고
                    num_correct = (outputs.argmax(-1) == targets).sum().item()
                    # 배치별 누적 전체 건수와 누적 전체 num_correct 건수로 accuracy 계산  
                    num_total += inputs.shape[0]
                    accu_num_correct += num_correct
                    accuracy = accu_num_correct / num_total
                    
                    #tqdm progress_bar에 진행 상황 및 running 평균 loss와 정확도 표시
                    progress_bar.update(1)
                    if batch_idx % 20 == 0 or (batch_idx + 1) == progress_bar.total:  # 20 batch 횟수마다 또는 맨 마지막 batch에서 update
                        progress_bar.set_postfix({"Loss": running_avg_loss,
                                                  "Accuracy": accuracy})
        # scheduler에 검증 데이터 기반에서 epoch레벨로 계산된 loss를 입력해줌.
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(running_avg_loss)
            self.current_lr = self.scheduler.get_last_lr()[0]

        return running_avg_loss, accuracy

    def fit(self, epochs):
        # epoch 시마다 학습/검증 결과를 기록하는 history dict 생성. learning rate 추가
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate_epoch(epoch)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f} Train Accuracy: {train_acc:.4f}",
                  f", Val Loss: {val_loss:.4f} Val Accuracy: {val_acc:.4f}" if val_loss is not None else "",
                  f", Current lr:{self.current_lr:.6f}")
            # epoch 시마다 학습/검증 결과를 기록. learning rate 추가
            history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)
            history['lr'].append(self.current_lr)

            # 만약 callbacks가 생성 인자로 들어온 다면 아래 수행. 만약 early stop 되어야 하면 is_epoch_loop_break로 for loop break
            if self.callbacks:
                is_epoch_loop_break = self._execute_callbacks(self.callbacks, self.model, epoch, val_loss, val_acc)
                if is_epoch_loop_break:
                    break
                                
        return history

    # 생성 인자로 들어온 callbacks list을 하나씩 꺼내서 ModelCheckpoint, EarlyStopping을 수행. 
    # EarlyStopping 호출 시 early stop 여부를 판단하는 is_early_stopped 반환
    def _execute_callbacks(self, callbacks, model, epoch, val_loss, val_acc):
        is_early_stopped = False
        
        for callback in self.callbacks:
            if isinstance(callback, ModelCheckpoint):
                if callback.monitor == 'val_loss':    
                    callback.save(model, epoch, val_loss)
                elif callback.monitor == 'val_acc':
                    callback.save(model, epoch, val_acc)
            if isinstance(callback, EarlyStopping):
                if callback.monitor == 'val_loss':
                    is_early_stopped = callback.check_early_stop(val_loss)
                if callback.monitor == 'val_acc':
                    is_early_stopped = callback.check_early_stop(val_acc)
                
        return is_early_stopped

    # 학습이 완료된 모델을 return
    def get_trained_model(self):
        return self.model
    
class Predictor:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    def evaluate(self, loader):
        self.model.eval()
        eval_metric = 0.0
        # 정확도 계산을 위한 전체 건수 및 누적 정확건수
        num_total = 0.0
        accu_num_correct = 0.0

        with tqdm(total=len(loader), desc=f"[Evaluating]", leave=True) as progress_bar:
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(loader):
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    pred = self.model(inputs)

                    # 정확도 계산을 위해 누적 전체 건수와 누적 전체 num_correct 건수 계산  
                    num_correct = (pred.argmax(-1) == targets).sum().item()
                    num_total += inputs.shape[0]
                    accu_num_correct += num_correct
                    eval_metric = accu_num_correct / num_total

                    progress_bar.update(1)
                    if batch_idx % 20 == 0 or (batch_idx + 1) == progress_bar.total:
                        progress_bar.set_postfix({"Accuracy": eval_metric})
        
        return eval_metric

    def predict_proba(self, inputs):
        self.model.eval()
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            #예측값을 반환하므로 targets은 필요 없음.
            #targets = targets.to(self.device)
            pred_proba = F.softmax(outputs, dim=-1) #또는 dim=1

        return pred_proba

    def predict(self, inputs):
        pred_proba = self.predict_proba(inputs)
        pred_class = torch.argmax(pred_proba, dim=-1)

        return pred_class

class EarlyStopping:
    def __init__(self, monitor='val_loss', mode='min', early_patience=5, verbose=1):
        self.monitor = monitor
        self.mode = mode
        self.early_patience = early_patience
        self.verbose = verbose
        self.best_value = float('inf') if mode == 'min' else -float('inf')
        self.counter = 0

    def is_improvement(self, value):
        if self.mode == 'min':
            return value < self.best_value
        else:
            return value > self.best_value

    def check_early_stop(self, value):
        is_early_stopped = False
        
        if self.is_improvement(value):
            self.best_value = value
            self.counter = 0
            is_early_stopped =False
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.early_patience}")
            if self.counter >= self.early_patience:
                is_early_stopped = True
                if self.verbose:
                    print("Early stopping happens and train stops")
        
        return is_early_stopped
    
    import os


class ModelCheckpoint:
    def __init__(self, checkpoint_dir='checkpoints', monitor='val_loss', mode='min', save_interval=1, verbose=1):
        self.checkpoint_dir = checkpoint_dir
        self.monitor = monitor
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else -float('inf')
        self.verbose = verbose
        self.save_interval = save_interval
        self._make_checkpoint_dir_unless()

    def _make_checkpoint_dir_unless(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
    
    # mode 유형에 따라 metric value값이 이전 epoch시 보다 향상 되었는지 확인하여 True/False 값 return
    def is_improvement(self, value):
        if self.mode == 'min':
            return value < self.best_value
        else:
            return value > self.best_value

    # self.best_value값 update, is_improvement() 반환값이 True인 경우만 수행. 
    def update_best_value(self, value):
        self.best_value = value

    def save(self, model, epoch, value):
        if self.save_interval == 1:
            if self.is_improvement(value):
                self._checkpoint_save(model, epoch, value)
                self.update_best_value(value)
            
        elif self.save_interval > 1:
            if (epoch + 1) % self.save_interval == 0:
                self._checkpoint_save(model, epoch, value)
                 
        # 수행하지 말고 참조만 할 것(save_interval 횟수마다 model 성능이 향상되는 경우 저장)
        # if (epoch + 1) % self.save_interval == 0 and self.is_improvement(value):
        #     self.update_best_value(value)
        #     self._checkpoint_save(model, epoch, value)
            
    def _checkpoint_save(self, model, epoch, value):
        checkpoint_path = os.path.join(self.checkpoint_dir, 
                                       f'checkpoint_epoch_{epoch+1}_{self.monitor}_{value:.4f}.pt')
        torch.save(model.state_dict(), checkpoint_path)
        if self.verbose:
            print(f"Saved model checkpoint at {checkpoint_path}")
