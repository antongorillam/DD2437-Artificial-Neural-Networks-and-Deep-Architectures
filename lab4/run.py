from util import *
from rbm import RestrictedBoltzmannMachine 
from dbn import DeepBeliefNet
import pandas as pd

if __name__ == "__main__":

    image_size = [28,28]

    train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

    ''' restricted boltzmann machine '''
    
    # print ("\nStarting a Restricted Boltzmann Machine..")
    # ndim_hidden = [500, 400, 300, 200]
    # n_iterations = 200
    

    # for hidden in ndim_hidden:
        
    #     rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
    #                                     ndim_hidden=hidden,
    #                                     is_bottom=True,
    #                                     image_size=image_size,
    #                                     is_top=False,
    #                                     n_labels=10,
    #                                     batch_size=10,
    #     )

    #     recon_losses = rbm.cd1(visible_trainset=train_imgs, n_iterations=n_iterations)
    #     df = pd.DataFrame(recon_losses, columns=['num_iterations', 'recon_losses'])
    #     df.to_csv(f'recon_losses_nHidden_{hidden}_n_iterations_{n_iterations}.csv')
    
    ''' deep- belief net '''
    dbn = DeepBeliefNet(sizes={"vis":image_size[0]*image_size[1], "hid":500, "pen":500, "top":2000, "lbl":10},
                        image_size=image_size,
                        n_labels=10,
                        batch_size=10
    )
    
    ''' greedy layer-wise training '''

    dbn.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=50)

    #dbn.recognize(train_imgs, train_lbls)
    
    #dbn.recognize(test_imgs, test_lbls)

    for digit in range(10):
        digit_1hot = np.zeros(shape=(1,10))
        digit_1hot[0,digit] = 1
        dbn.generate(digit_1hot, name="rbms")

    ''' fine-tune wake-sleep training '''

    # dbn.train_wakesleep_finetune(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=10)

    # dbn.recognize(train_imgs, train_lbls)
    
    # dbn.recognize(test_imgs, test_lbls)
    
    # for digit in range(10):
    #     digit_1hot = np.zeros(shape=(1,10))
    #     digit_1hot[0,digit] = 1
    #     dbn.generate(digit_1hot, name="dbn")
