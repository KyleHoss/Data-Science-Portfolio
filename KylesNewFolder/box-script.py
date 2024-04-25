htmlFiles = ['html1','html2','html3']
townName = ['kirkwood','this','testing3']

for i in range(0,len(htmlFiles)):

    file1 = open("myfile.txt", "a")
 
    # writing newline character
    file1.write(f'''\n<!-- {townName[i]} -->
    <a href="{htmlFiles[i]}">
        <div class="stl-port-box">
            <img src="../Heart-Failure/HeartFailure.jpg" alt="Heart Failure">
            <div class="stl-port-layer">
                <h4>{townName[i]}</h4>
            </div>
        </div>
    </a>''')

file1.close()
