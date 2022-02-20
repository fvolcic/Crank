#include <iostream>
#include <sstream>
#include <string> 
#include <cmath>

using namespace std; 

int main(){

    double x = nan(""); 
    x = 3.5; 
    stringstream ss;

    ss << x; 

    string s = ss.str(); 

    cout << s.size() << endl;
    cout << s << endl;   

    if(s[0] == 'n'){
        cout << "ERROR" << endl; 
    }
}