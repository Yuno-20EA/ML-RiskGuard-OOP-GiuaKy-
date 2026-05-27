2.	#include <iostream>
3.	#include <cassert>
4.	
5.	int sum(int a, int b) {
6.	    return a + b; 
7.	}
8.	
9.	void test_sum_positive_numbers() {
10.	    assert(sum(2, 3) == 5);
11.	}
12.	
13.	void test_sum_negative_numbers() {
14.	    assert(sum(-1, -1) == -2);
15.	}
16.	
17.	void test_sum_zero() {
18.	    assert(sum(0, 5) == 5);
19.	}
20.	
21.	    std::cout << "Dang chay kiem tra tu dong...\n";
22.	
23.	    test_sum_positive_numbers();
24.	    test_sum_negative_numbers();
25.	    test_sum_zero();
26.	
27.	    std::cout << "Chuc mung! Tat ca cac bai test deu THANH CONG!\n";
28.	    return 0;
29.	}
