import torch
import torch.nn as nn
import pdb


class ValueGuide(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, cond, t):
        output = self.model(x, cond, t)
        return output.squeeze(dim=-1)

    def gradients(self, x, *args):
        x.requires_grad_()
        y = self(x, *args)
        grad = torch.autograd.grad([y.sum()], [x])[0]
        x.detach()
        return y, grad


class GoalValueGuide(nn.Module):
    
        def __init__(self, model):
            super().__init__()
            self.model = model # Q(s, g)

        def set_goal(self, goal):
            self.goal = goal
            self.goal_dim = goal.shape[-1]
    
        def forward(self, x, cond, t, ):
            # input: last state + goal
            # x = torch.cat(x[0], x[-1], dim=-1)
            output = self.model(x, cond, t)
            return output.squeeze(dim=-1)

        def gradients(self, x, *args):
            # only calculate gradients to the last state
            s = x[:, -1:, :].clone()
            action_dim = s.shape[-1] - self.goal_dim
            s[:, :, :action_dim] = 0 # set actions to 0

            s.requires_grad_()
            
            # make goals have same first dimension with s
            goals = self.goal.repeat(s.shape[0], s.shape[1],  1)
            x_q = torch.cat([s, goals], dim=1)
            # x_q = torch.tensor(x_q, dtype=torch.float32)

            y = self(x_q, *args)

            # grad = torch.autograd.grad([y.sum()], [s])[0]
            y.backward()
            grad = s.grad
            s.detach()

            # change it back to the shape of x
            grad_output = torch.zeros_like(x, device=x.device)
            grad_output[:, -1:, :] = grad

            return y, grad_output

            # x.requires_grad_()
            # y = self(x, *args)
            # grad = torch.autograd.grad([y.sum()], [x])[0]
            # x.detach()
            # return y, grad

            