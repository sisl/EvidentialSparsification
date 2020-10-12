from dst import *
import numpy as np

'''Write test cases.'''
class TestCases:
    def __init__(self, J = 3, K = 2, N = 1):

        self.J = J
        self.K = K
        self.N = N

        self.weights = np.reshape(np.flip(np.arange(J * K)), (J, K))
        self.bias = np.ones((K, 1))
        self.bias[0,:] = 3.
        self.features = np.reshape(np.arange(N * J), (N, J))

        print("beta hat", self.weights)
        print("bias", self.bias)
        print("phi", self.features)

        # initialize a dst instance
        self.dst_test = DST()
        self.dst_test.weights_from_linear_layer(self.weights, self.bias, self.features)
        self.dst_test.get_output_mass(num_classes = K)

    '''Test cases for: weights from linear layer.'''

    # test beta weights for correctness and shape
    def test_weights_1(self): 
        assert self.dst_test.beta.shape == (self.J, self.K), "Incorrect beta star shape!"

        correct_beta = np.ones((self.J, self.K))
        correct_beta[:, 0] = 0.5
        correct_beta[:, 1] = -0.5
        assert np.array_equal(self.dst_test.beta, correct_beta), "Incorrect beta star array values!"
        print("Test Weights 1 Completed.")

    # test alpha weights for correctness and shape
    def test_weights_2(self):
        assert self.dst_test.alpha.shape == (self.J, self.K), "Incorrect alpha shape!"

        correct_alpha = np.ones((self.J, self.K))
        correct_alpha[0, 0] = 2.5 / 3.
        correct_alpha[1, 0] = 2.5 / 3. - 0.5
        correct_alpha[2, 0] = 2.5 / 3. - 1.
        correct_alpha[0, 1] = - 2.5 / 3.
        correct_alpha[1, 1] = - 2.5 / 3. + 0.5
        correct_alpha[2, 1] = - 2.5 / 3. + 1.
        assert np.allclose(self.dst_test.alpha, correct_alpha, rtol=1e-16, atol=1e-16, equal_nan=False), "Incorrect alpha array values!"
        print("Test Weights 2 Completed.")

    # test evidential weights for correctness and shape
    def test_weights_3(self):
        assert self.dst_test.evidential_weights.shape == (self.N, self.J, self.K), "Incorrect evidential weights shape!"

        correct_ev_weights = np.ones((self.N, self.J, self.K))
        correct_ev_weights[:, 0, 0] = 0.5 * 0. + 5. / 6. 
        correct_ev_weights[:, 1, 0] = 0.5 * 1. + 1. / 3.
        correct_ev_weights[:, 2, 0] = 0.5 * 2. - 1. / 6.
        correct_ev_weights[:, 0, 1] = - 0.5 * 0. - 5. / 6. 
        correct_ev_weights[:, 1, 1] = - 0.5 * 1. - 1. / 3.
        correct_ev_weights[:, 2, 1] = - 0.5 * 2. + 1. / 6.
        assert np.allclose(self.dst_test.evidential_weights, correct_ev_weights, rtol=1e-16, atol=1e-16, equal_nan=False), "Incorrect evidential weights array values!"
        print("Test Weights 3 Completed.")

    # test pos/neg evidential weights for correctness and shape
    # - make sure subtract up to evidential weights
    # - make sure only one element is non-zero out of pair
    def test_weights_4(self):
        assert self.dst_test.evidential_weights_pos_jk.shape == (self.N, self.J, self.K), "Incorrect pos shape!"
        assert self.dst_test.evidential_weights_neg_jk.shape == (self.N, self.J, self.K), "Incorrect neg shape!"

        assert np.allclose(self.dst_test.evidential_weights_pos_jk - self.dst_test.evidential_weights_neg_jk, self.dst_test.evidential_weights, rtol=1e-16, atol=1e-16, equal_nan=False), "Incorrect pos/neg breakdown!"
        assert np.all(np.logical_or((self.dst_test.evidential_weights_pos_jk == 0), (self.dst_test.evidential_weights_neg_jk == 0))), "Incorrect pos/neg breakdown (zeros)!"
        print("Test Weights 4 Completed.")

    # test pos_k/neg_k weights for correctness and shape
    def test_weights_5(self):
        assert self.dst_test.evidential_weights_pos_k.shape == (self.N, self.K), "Incorrect pos_k shape!"
        assert self.dst_test.evidential_weights_neg_k.shape == (self.N, self.K), "Incorrect neg_k shape!"

        correct_ev_k_pos = np.zeros((self.N, self.K))
        correct_ev_k_pos[:, 0] = 2.5

        correct_ev_k_neg = np.zeros((self.N, self.K))
        correct_ev_k_neg[:, 1] = 2.5

        assert np.allclose(self.dst_test.evidential_weights_pos_k, correct_ev_k_pos, rtol=1e-16, atol=1e-16, equal_nan=False), "Incorrect pos_k array values!"
        assert np.allclose(self.dst_test.evidential_weights_neg_k, correct_ev_k_neg, rtol=1e-16, atol=1e-16, equal_nan=False), "Incorrect neg_k array values!"

        print("Test Weights 5 Completed.")

    '''Test cases for: get_output_mass.'''

    # test powerset: len and visually
    def test_masses_1(self):
        assert len(self.dst_test.powerset) == 2**self.K, "Incorrect powerset length!"
        assert self.dst_test.powerset == [(), (0,), (1,), (0,1)], "Incorrect powerset!"

        print("Test Masses 1 Completed.")         

    # test output_mass: 
    # - len of keys 
    # - no singletons
    # - completeness of keys
    # - correctness on a 2 or 3 class example
    def test_masses_2(self):

        C = 0 + (np.exp(2.5) - np.exp(-2.5)) + np.exp(-2.5)

        correct_mass = np.exp(-2.5) / C 

        assert len(self.dst_test.output_mass) == 1, "Incorrect number of non-singleton masses!"
        assert self.dst_test.output_mass[tuple((0,1))] == correct_mass, "Incorrect non-singleton mass value!"

        print("Test Masses 2 Completed.")

    # test ouput_mass_singleton:
    # - right shape
    # - completeness of keys
    # - correctness on a 2 or 3 class example
    def test_masses_3(self):

        C = 0 + (np.exp(2.5) - np.exp(-2.5)) + np.exp(-2.5)
        
        correct_mass_singletons = np.zeros((self.N, self.K))
        correct_mass_singletons[:, 0] = (np.exp(2.5) - np.exp(-2.5))
        correct_mass_singletons /= C

        assert self.dst_test.output_mass_singletons.shape == (self.N, self.K), "Incorrect shape for singleton masses!"
        assert np.allclose(self.dst_test.output_mass_singletons, correct_mass_singletons, rtol=1e-16, atol=1e-16, equal_nan=False), "Incorrect singleton mass array values!"

        print("Test Masses 3 Completed.")

'''Write test cases.'''
class TestCasesLarger:
    def __init__(self, J = 3, K = 3, N = 1):

        self.J = J
        self.K = K
        self.N = N

        self.weights = np.reshape(np.flip(np.arange(J * K)), (J, K))
        self.bias = np.ones((K, 1))
        self.bias[0,:] = -1.
        self.bias[2,:] = 3.
        self.features = np.reshape(np.arange(N * J), (N, J))

        print("beta hat", self.weights)
        print("bias", self.bias)
        print("phi", self.features)

        # initialize a dst instance
        self.dst_test = DST()
        self.dst_test.weights_from_linear_layer(self.weights, self.bias, self.features)
        self.dst_test.get_output_mass(num_classes = K)

    '''Test cases for: weights from linear layer.'''

    # test beta weights for correctness and shape
    def test_weights_1(self): 
        assert self.dst_test.beta.shape == (self.J, self.K), "Incorrect beta star shape!"

        correct_beta = np.ones((self.J, self.K))
        correct_beta[:, 1] = 0.
        correct_beta[:, 2] = -1.
        assert np.array_equal(self.dst_test.beta, correct_beta), "Incorrect beta star array values!"
        print("Test Weights 1 Completed.")

    # test alpha weights for correctness and shape
    def test_weights_2(self):
        assert self.dst_test.alpha.shape == (self.J, self.K), "Incorrect alpha shape!"

        correct_alpha = np.zeros((self.N, self.J, self.K))
        correct_alpha[:, 0, 0] = 1./3.
        correct_alpha[:, 0, 2] = -1./3.
        correct_alpha[:, 1, 0] = -2./3.
        correct_alpha[:, 1, 2] = 2./3.
        correct_alpha[:, 2, 0] = -5./3.
        correct_alpha[:, 2, 2] = 5./3.
        assert np.allclose(self.dst_test.alpha, correct_alpha, rtol=1e-16, atol=1e-16, equal_nan=False), "Incorrect alpha array values!"
        print("Test Weights 2 Completed.")

    # test evidential weights for correctness and shape
    def test_weights_3(self):
        assert self.dst_test.evidential_weights.shape == (self.N, self.J, self.K), "Incorrect evidential weights shape!"

        correct_ev_weights = np.zeros((self.N, self.J, self.K))
        correct_ev_weights[:, :, 0] = 1./3.
        correct_ev_weights[:, :, 2] = -1./3.    
        assert np.allclose(self.dst_test.evidential_weights, correct_ev_weights, rtol=1e-16, atol=1e-16, equal_nan=False), "Incorrect evidential weights array values!"
        print("Test Weights 3 Completed.")

    # test pos/neg evidential weights for correctness and shape
    # - make sure subtract up to evidential weights
    # - make sure only one element is non-zero out of pair
    def test_weights_4(self):
        assert self.dst_test.evidential_weights_pos_jk.shape == (self.N, self.J, self.K), "Incorrect pos shape!"
        assert self.dst_test.evidential_weights_neg_jk.shape == (self.N, self.J, self.K), "Incorrect neg shape!"

        assert np.allclose(self.dst_test.evidential_weights_pos_jk - self.dst_test.evidential_weights_neg_jk, self.dst_test.evidential_weights, rtol=1e-16, atol=1e-16, equal_nan=False), "Incorrect pos/neg breakdown!"
        assert np.all(np.logical_or((self.dst_test.evidential_weights_pos_jk == 0), (self.dst_test.evidential_weights_neg_jk == 0))), "Incorrect pos/neg breakdown (zeros)!"
        print("Test Weights 4 Completed.")

    # test pos_k/neg_k weights for correctness and shape
    def test_weights_5(self):
        assert self.dst_test.evidential_weights_pos_k.shape == (self.N, self.K), "Incorrect pos_k shape!"
        assert self.dst_test.evidential_weights_neg_k.shape == (self.N, self.K), "Incorrect neg_k shape!"

        correct_ev_k_pos = np.zeros((self.N, self.K))
        correct_ev_k_pos[:, 0] = 1.

        correct_ev_k_neg = np.zeros((self.N, self.K))
        correct_ev_k_neg[:, 2] = 1.

        assert np.allclose(self.dst_test.evidential_weights_pos_k, correct_ev_k_pos, rtol=1e-15, atol=1e-15, equal_nan=False), "Incorrect pos_k array values!"
        assert np.allclose(self.dst_test.evidential_weights_neg_k, correct_ev_k_neg, rtol=1e-15, atol=1e-15, equal_nan=False), "Incorrect neg_k array values!"

        print("Test Weights 5 Completed.")

    '''Test cases for: get_output_mass.'''

    # test powerset: len and visually
    def test_masses_1(self):
        assert len(self.dst_test.powerset) == 2**self.K, "Incorrect powerset length!"
        assert self.dst_test.powerset == [(), (0,), (1,), (2,), (0,1), (0,2), (1,2), (0,1,2)], "Incorrect powerset!"

        print("Test Masses 1 Completed.")         

    # test output_mass: 
    # - len of keys 
    # - no singletons
    # - completeness of keys
    # - correctness on a 2 or 3 class example
    def test_masses_2(self):

        C = np.exp(1) - 1. + (1-np.exp(-1)) + np.exp(-1)
        print(self.dst_test.output_mass[tuple((0,1))], (1 - np.exp(-1))/1./C)

        assert len(self.dst_test.output_mass) == 4, "Incorrect number of non-singleton masses!"
        assert np.allclose(self.dst_test.output_mass[tuple((0,1))], (1 - np.exp(-1))/1./C, rtol=1e-15, atol=1e-15, equal_nan=False), "Incorrect non-singleton mass value!"
        assert self.dst_test.output_mass[tuple((0,2))] == 0, "Incorrect non-singleton mass value!"
        assert self.dst_test.output_mass[tuple((1,2))] == 0, "Incorrect non-singleton mass value!"
        assert np.allclose(self.dst_test.output_mass[tuple((0,1,2))], np.exp(-1)/1./C, rtol=1e-15, atol=1e-15, equal_nan=False), "Incorrect non-singleton mass value!"

        print("Test Masses 2 Completed.")

    # test ouput_mass_singleton:
    # - right shape
    # - completeness of keys
    # - correctness on a 2 or 3 class example
    def test_masses_3(self):

        C = np.exp(1) - 1. + (1-np.exp(-1)) + np.exp(-1)
        
        correct_mass_singletons = np.zeros((self.N, self.K))
        correct_mass_singletons[:,0] = np.exp(1) - 1.
        correct_mass_singletons /= C

        assert self.dst_test.output_mass_singletons.shape == (self.N, self.K), "Incorrect shape for singleton masses!"
        assert np.allclose(self.dst_test.output_mass_singletons, correct_mass_singletons, rtol=1e-16, atol=1e-16, equal_nan=False), "Incorrect singleton mass array values!"

        print("Test Masses 3 Completed.")

# Create some test cases
def main():
    
    test = TestCases()

    test.test_weights_1()
    test.test_weights_2()
    test.test_weights_3()
    test.test_weights_4()
    test.test_weights_5()

    test.test_masses_1()
    test.test_masses_2()
    test.test_masses_3()

    test2 = TestCasesLarger()

    test2.test_weights_1()
    test2.test_weights_2()
    test2.test_weights_3()
    test2.test_weights_4()
    test2.test_weights_5()

    test2.test_masses_1()
    test2.test_masses_2()
    test2.test_masses_3()

    J = 10
    K = 3
    N = 1

    weights = np.random.rand(J, K)
    bias = np.random.rand(K, 1)
    features = np.random.rand(N, J)

    dst_obj = DST()
    dst_obj.weights_from_linear_layer(weights, bias, features)
    dst_obj.get_output_mass(num_classes = K)

    print(dst_obj.powerset)
    print(dst_obj.output_mass_singletons)
    print(dst_obj.output_mass)
  
if __name__== "__main__":
    main()
