from os import system

print("find bin/ -type f -executable -exec rm {} +")
system("find bin/ -type f -executable -exec rm {} +")