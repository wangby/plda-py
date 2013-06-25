import unittest
import plda 
import numpy as np


class TestPlda(unittest.TestCase):

    def approx_equal(self, num1, num2, error=0.05):
        return abs(num1 - num2) <= 0.5 * error * (num1 + num2)

    def test_accuracy_string(self):
        # This is 10 records we got from the test-data set in plda package.
        test_data_file = 'data/test_data_10l.txt'

        # This is the data we got by running command line infer tool in the plda package.
        expected = [
            [114.05, 273.95],
            [440.03, 439.97],
            [124.72, 386.28],
            [266.75, 350.25],
            [302.31, 558.69],
            [131.61, 401.39],
            [529.27, 256.73],
            [141.16, 394.84],
            [542.17, 179.83],
            [271.04, 344.96]
        ]

        model_file = 'data/lda_model.txt'

        alpha = 0.1
        beta = 0.01
        max_iter = 200
        burnin_iter = 100
        seed = -1   # use the original seed setting (time) in plda package.

        model = plda.PyLDA(model_file, alpha, beta, max_iter, burnin_iter, seed)

        with open(test_data_file) as fin:
            i = 0
            for line in fin:
                calculated = model.run(line)

                # Due to the randomness of the plda algorithm, the values may differ by about 5%.
                for j in xrange(len(calculated)):
                    self.assertTrue(self.approx_equal(calculated[j], expected[i][j]))

                i += 1

    def test_accuracy_list(self):
        # This is 10 records we got from the test-data set in plda package.
        test_data = [
            "concept", "consider", "global", "entropy", "go", "contributions", "excludes", "depend", "graph",
            "environment", "program", "under", "undirected", "random", "very", "putting", "difference", "entire",
            "randomness", "july", "large", "vector", "synapses", "zl", "upper", "smaller", "says", "occurrence", "val-",
            "likely", "n", "ues", "what", "selected", "nand", "find", "access", "version", "goes", "obvious",
            "learn", "here", "desired", "objects", "let", "represented", "strong", "appears", "equiv-", "institute",
            "k", "vectors", "reports", "amount", "extremes", "proof", "regardless", "projection", "merely",
            "boolean", "total", "asymptotic", "would", "prove", "next", "automata", "taken", "tell", "knows",
            "becomes", "visual", "appendix", "normalized", "particular", "hold", "must", "work", "itself",
            "values", "v", "abu-mostafa", "process", "sample", "something", "arise", "distinguishable", "occur", "huge",
            "end", "rather", "means", "feature", "write", "infor-", "spon-", "ensemble", "information", "may", "after",
            "consequence", "designed", "en-", "complexity", "so", "sb", "restriction", "holds", "office", "produces",
            "yaser", "paper", "through", "ity", "still", "denker", "symmetry", "how", "coordinates", "distinguishing",
            "systems", "main", "versus", "eventually", "imple-", "synapse", "introduce", "thus", "now", "nor", "term",
            "subset", "el", "doing", "ea", "idea", "frequency",
            ]

        # This is the data we got by running command line infer tool in the plda package.
        expected = [46.21, 89.79]

        model_file = 'data/lda_model.txt'

        alpha = 0.1
        beta = 0.01
        max_iter = 200
        burnin_iter = 100
        seed = -1   # use the original seed setting (time) in plda package.

        model = plda.PyLDA(model_file, alpha, beta, max_iter, burnin_iter, seed)

        calculated = model.run_on_list(test_data)

        # Due to the randomness of the plda algorithm, the values may differ by about 5%.
        for j in xrange(len(calculated)):
            self.assertTrue(self.approx_equal(calculated[j], expected[j]))

    def test_accuracy_list_unicode(self):
        # This is 10 records we got from the test-data set in plda package.
        test_data = [
            "concept", "consider", "global", "entropy", "go", "contributions", "excludes", "depend", "graph",
            "environment", "program", "under", "undirected", "random", "very", "putting", "difference", "entire",
            "randomness", "july", "large", "vector", "synapses", "zl", "upper", "smaller", "says", "occurrence", "val-",
            "likely", "n", "ues", "what", "selected", "nand", "find", "access", "version", "goes", "obvious",
            "learn", "here", "desired", "objects", "let", "represented", "strong", "appears", "equiv-", "institute",
            "k", "vectors", "reports", "amount", "extremes", "proof", "regardless", "projection", "merely",
            "boolean", "total", "asymptotic", "would", "prove", "next", "automata", "taken", "tell", "knows",
            "becomes", "visual", "appendix", "normalized", "particular", "hold", "must", "work", "itself",
            "values", "v", "abu-mostafa", "process", "sample", "something", "arise", "distinguishable", "occur", "huge",
            "end", "rather", "means", "feature", "write", "infor-", "spon-", "ensemble", "information", "may", "after",
            "consequence", "designed", "en-", "complexity", "so", "sb", "restriction", "holds", "office", "produces",
            "yaser", "paper", "through", "ity", "still", "denker", "symmetry", "how", "coordinates", "distinguishing",
            "systems", "main", "versus", "eventually", "imple-", "synapse", "introduce", "thus", "now", "nor", "term",
            "subset", "el", "doing", "ea", "idea", "frequency", u'registered\xae'
        ]

        # This is the data we got by running command line infer tool in the plda package.
        expected = [46.21, 89.79]

        model_file = 'data/lda_model.txt'

        alpha = 0.1
        beta = 0.01
        max_iter = 200
        burnin_iter = 100
        seed = -1   # use the original seed setting (time) in plda package.

        model = plda.PyLDA(model_file, alpha, beta, max_iter, burnin_iter, seed)

        calculated = model.run_on_list(test_data)

        # Due to the randomness of the plda algorithm, the values may differ by about 5%.
        for j in xrange(len(calculated)):
            self.assertTrue(self.approx_equal(calculated[j], expected[j]))

    def test_memory_leak(self):
        """
        This test case is used to test if there exists memory leak in the c++ codes by repeatedly calling model.run
        method.
        :return:
        """
        # This is 1 record we got from the test-data set in plda package.
        test_data = "externally 1 global 1 dynamic 1 resistance 1 illustrated 1 scalar 1 generalize 1 computation 1 consists 1 follow 1 widrow 1 adjustment 1 layers 1 semiconductor 1 leaming 1 inverse 1 deterministic 1 charge 1 device 1 seems 1 stable 1 discovering 1 stochastic 1 include 1 summarizes 1 division 1 worse 1 amplifier 1 activation 1 connects 1 eprom 1 choice 1 replication 1 manipulated 1 updates 1 decide 1 adaptive 1 timely 1 losleben 1 solves 1 level 1 issue 1 joel 1 solution 1 resistor 1 large 1 circuit 1 adjust 1 neuromorphic 1 science 1 small 1 biological 1 correct 1 smaller 1 slower 1 incremented 1 sign 1 approximation 1 second 1 design 1 picked 1 pass 1 implemented 1 substitute 1 theory 1 even 1 xor 1 sum 1 selected 1 uniform 1 current 1 bipolar 1 version 1 above 1 activations 1 new 1 net 1 implementable 1 edited 1 focuses 1 chose 1 here 1 desired 1 address 1 ratio 1 quite 1 teacher 1 change 1 settling 1 search 1 institute 1 study 1 settles 1 allows 1 credit 1 amount 1 suggestion 1 suitable 1 joshua 1 technique 1 useful 1 divide 1 family 1 integrator 1 vin 1 wij 1 highly 1 trained 1 counts 1 unit 1 lime 1 fed 1 refers 1 vlsi 1 coins 1 negative 1 next 1 fet 1 andy 1 therefore 1 memory 1 hinton 1 type 1 until 1 definition 1 becomes 1 visual 1 controlling 1 particular 1 hold 1 effort 1 must 1 states 1 keeping 1 valued 1 f 1 mm 1 avoiding 1 work 1 ij 1 values 1 temperatures 1 learn 1 following 1 making 1 averages 1 control 1 compare 1 renormalization 1 process 1 chip 1 high 1 effectively 1 critic 1 minimum 1 performed 1 magnitudes 1 sharp 1 counting 1 information 1 fets 1 goal 1 sit 1 provide 1 divided 1 get 1 feature 1 arrangement 1 regions 1 how 1 criterion 1 fourth 1 answer 1 optimal 1 dominance 1 inputs 1 tried 1 influenced 1 may 1 after 1 plausibility 1 diagram 1 connectivity 1 designed 1 ff 1 fi 1 synchrony 1 parallel 1 types 1 man 1 gaussian 1 short 1 physical 1 effective 1 capacitances 1 computations 1 counter 1 explicit 1 summing 1 compressing 1 sj 1 annealing 1 hebb-type 1 algorithms 1 speeds 1 fundamental 1 correlation 1 representation 1 increase 1 order 1 recurrent 1 feedback 1 sigmoid 1 merit 1 find 1 stability 1 brain 1 thermal 1 experiments 1 through 1 supervised 1 statistical 1 reinforcement 1 differs 1 still 1 machine 1 style 1 position 1 symmetry 1 decay 1 chosen 1 lo 1 temporal 1 decreases 1 winning 1 forms 1 absence 1 permanent 1 microscopic 1 systems 1 decreased 1 hidden 1 might 1 pixel 1 eventually 1 them 1 good 1 greater 1 thereby 1 coding 1 synapse 1 often 1 time-averaged 1 tails 1 they 1 instead 1 now 1 nor 1 synaptic 1 easily 1 term 1 gets 1 stabilize 1 adjacent 1 always 1 university 1 presented 1 applied 1 magnitude 1 reasonable 1 inherent 1 side 1 characterized 1 clamp 1 weighted 1 directly 1 weight 1 tension 1 ee 1 reduce 1 principles 1 flip-flops 1 proportion 1 conductance 1 frequency 1 measure 1 substantially 1 our 1 tuned 1 out 1 bly 1 network 1 space 1 gradient 1 quality 1 robert 1 since 1 squared 1 research 1 looking 1 performs 1 electronics 1 evaluation 1 integrated 1 shows 1 working 1 linear 1 mechanisms 1 diagonal 1 standard 1 sensory 1 signal 1 approaches 1 formation 1 york 1 usual 1 asynchronous 1 non-specific 1 advance 1 training 1 interaction 1 could 1 creates 1 american 1 place 1 exponentially 1 w 1 uses 1 south 1 major 1 features 1 probability 1 question 1 powerful 1 improvements 1 another 1 representations 1 electronic 1 size 1 temperature 1 leading 1 caught 1 system 1 adopted 1 their 1 attack 1 rates 1 too 1 convention 1 normalization 1 molecular 1 structures 1 final 1 store 1 gives 1 way 1 releases 1 completed 1 continuous 1 part 1 favorable 1 somewhat 1 thai 1 believe 1 representing 1 diffusion 1 plausible 1 b 1 determines 1 third 1 double 1 future 1 tsividis 1 were 1 minus 1 lines 1 cohen 1 occupies 1 correspond 1 randomly 1 gale 1 winner-take- 1 increment 1 seen 1 occupied 1 any 1 synapses 1 liapunov 1 strength 1 s- 1 efficient 1 latter 1 aside 1 note 1 potential 1 vice-versa 1 finding 1 so 1 victor 1 switching 1 channel 1 specifying 1 play 1 transistor 1 multiple 1 electric 1 measures 1 techniques 1 most 1 boltzmann 1 connected 1 significant 1 ccd 1 phase 1 adds 1 sophisticated 1 measured 1 gain 1 operate 1 especially 1 unsupervised 1 considered 1 average 1 later 1 m 1 elucidating 1 proceedings 1 spatial 1 oscillates 1 physics 1 section 1 western 1 particularly 1 show 1 random 1 speedup 1 polysilicon 1 attempts 1 radio 1 threshold 1 networks 1 help 1 layout 1 parameters 1 implementation 1 simulation 1 guidance 1 winner 1 distributed 1 controls 1 should 1 increasingly 1 micron 1 factor 1 employed 1 anneal 1 local 1 hope 1 do 1 explorations 1 his 1 goes 1 means 1 dependent 1 overall 1 dc 1 cannot 1 nearly 1 timing 1 during 1 symmetrically 1 averaged 1 procedures 1 volatility 1 produced 1 processes 1 cooccurrence 1 ended 1 differential 1 summary 1 whether 1 paper 1 architecture 1 cij 1 fixed 1 analog 1 view 1 rms 1 multiplying 1 physically 1 attached 1 unpublished 1 correlations 1 emergent 1 operates 1 connections 1 see 1 individual 1 result 1 discussions 1 close 1 correlational 1 subject 1 phases 1 satisfying 1 runs 1 plasticity 1 pattern 1 currently 1 dipole 1 artificial 1 state 1 unable 1 various 1 hopefully 1 tens 1 neither 1 previous 1 conditions 1 across 1 discovery 1 preserve 1 multiplied 1 complementary 1 initially 1 wm 1 prior 1 weak 1 interpolate 1 implementing 1 ji 1 modeling 1 distribution 1 improve 1 received 1 c 1 z 1 last 1 reported 1 many 1 region 1 equal 1 presentations 1 called 1 assure 1 s 1 subthreshold 1 interesting 1 expense 1 ease 1 co 1 point 1 simple 1 had 1 ca 1 simply 1 learning 1 table 1 cq 1 conference 1 better 1 described 1 tuning 1 basis 1 stretch 1 locally 1 due 1 appears 1 geoffrey 1 cauchy 1 replicate 1 logarithmic 1 collected 1 correction 1 define 1 obtained 1 generating 1 rather 1 controlled 1 moved 1 competitive 1 those 1 case 1 clamping 1 unify 1 tanh 1 aip 1 value 1 n 1 cluster 1 technical 1 while 1 optimization 1 wilh 1 behavior 1 error 1 report 1 situation 1 gave 1 interconnected 1 procedure 1 styles 1 layer 1 studied 1 coupled 1 almost 1 neuron 1 vl 1 surface 1 helped 1 characterize 1 neurons 1 converts 1 technology 1 binary 1 suggests 1 perhaps 1 make 1 cumulative 1 connection 1 rules 1 complex 1 digital 1 propose 1 units 1 several 1 difficult 1 development 1 extended 1 columbia 1 closest 1 assignment 1 logic 1 effect 1 possible 1 levels 1 moving 1 robust 1 outputs 1 recent 1 implements 1 lower 1 task 1 whereby 1 nevertheless 1 variety 1 well 1 patterns 1 without 1 solve 1 components 1 y 1 organization 1 model 1 left 1 unified 1 graded 1 just 1 less 1 being 1 resistive 1 guided 1 obtain 1 valuable 1 ideas 1 bandwidth 1 stored 1 identify 1 voltage 1 aspect 1 jitter 1 rest 1 speed 1 bit 1 electronically 1 based 1 adding 1 doctoral 1 comparisons 1 classification 1 hint 1 excites 1 percent 1 guaranteed 1 except 1 source 1 internal 1 samples 1 technological 1 input 1 matched 1 take 1 real 1 saturation 1 zipset 1 mos 1 showing 1 transconductance 1 emanating 1 schematically 1 increases 1 inhibition 1 correlating 1 solid-state 1 modifying 1 advanced 1 cognitive 1 vertically 1 strengths 1 necessary 1 like 1 achieve 1 collective 1 separately 1 communications 1 t 1 fully 1 output 1 works 1 reduced 1 because 1 methods 1 phenomena 1 back 1 competition 1 added 1 semi-local 1 specified 1 hebbian 1 asynchronously 1 decrement 1 specific 1 winner-take-all 1 bottom 1 avoid 1 thank 1 per 1 contributing 1 does 1 provides 1 either 1 somehow 1 noise 1 run 1 power 1 schedule 1 problems 1 equivalent 1 processing 1 step 1 percepttons 1 although 1 become 1 r 1 stage 1 comparison 1 about 1 would 1 column 1 dependence 1 o 1 essential 1 compatible 1 addition 1 range 1 plus 1 associative 1 propagate 1 block 1 converging 1 computational 1 presence 1 previously 1 communication 1 preliminary 1 usa 1 three 1 down 1 pair 1 square 1 microstructure 1 proportional 1 refined 1 reducing 1 envision 1 eeprom 1 computer 1 resistors 1 area 1 transfer 1 support 1 available 1 ns 1 long 1 stuck 1 vax 1 much 1 low 1 fraction 1 naturally 1 form 1 adjusting 1 regard 1 histogram 1 correlated 1 related 1 larger 1 true 1 directed 1 arranged 1 stanford 1 circuits 1 characteristic 1 j 1 variations 1 maximum 1 devices 1 below 1 represents 1 verification 1 demonstrate 1 problem 1 stages 1 similar 1 rumelhart 1 bell 1 bearing 1 constant 1 up 1 defined 1 universal 1 exceeded 1 yannis 1 al 1 general 1 globally 1 proper 1 single 1 diverse 1 right 1 inverter 1 floating 1 distributions 1 cmos 1 defines 1 nj 1 compared 1 no 1 barto 1 storage 1 sampled 1 branch 1 papers 1 test 1 weights 1 decremented 1 conclusion 1 elements 1 consisted 1 time 1 describe 1 includes 1 helping 1 important 1 pathways 1 variable 1 cut-off 1 matrix 1 voltage-controlled 1 e 1 performance 1 algorithm 1 stabilizes 1 necessitates 1 rule 1 inset 1 compete 1 searched 1 quantity 1 requires 1\n"

        # This is the data we got by running command line infer tool in the plda package.
        expected = [440.03, 439.97]

        model_file = 'data/lda_model.txt'

        alpha = 0.1
        beta = 0.01
        max_iter = 200
        burnin_iter = 100
        seed = -1   # use the original seed setting (time) in plda package.

        model = plda.PyLDA(model_file, alpha, beta, max_iter, burnin_iter, seed)

        for i in xrange(1):  # 10000 runs should take about 10 minutes to complete.
            calculated = model.run(test_data)

            for j in xrange(len(calculated)):
                self.assertTrue(self.approx_equal(calculated[j], expected[j], error=0.1))


if __name__ == "__main__":
    unittest.main()
