async function decompile() {
    const btn = document.getElementById('decompile-btn');
    const output = document.getElementById('output-code');
    
    output.textContent = "// 反编译中...";
    output.style.color = "#f8f8f2"; // Reset color
    btn.disabled = true;

    const arch = document.getElementById('arch').value;
    const machineCode = document.getElementById('machine-code').value.trim();

    if (!machineCode) {
        output.textContent = "// 错误: 请输入机器码.";
        output.style.color = "#ff6b6b";
        btn.disabled = false;
        return;
    }

    try {
        const response = await fetch('/decompile', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                arch: arch,
                machine_code: machineCode
            })
        });

        const data = await response.json();

        if (response.ok) {
            output.textContent = data.best_c_code || "// 生成 C 函数代码失败";
            output.style.color = "#f8f8f2";
        } else {
            output.textContent = `// 服务器错误:\n${data.detail || data.error || '未知错误'}`;
            output.style.color = "#ff6b6b";
        }
    } catch (error) {
        output.textContent = `// 网络错误:\n${error.message}`;
        output.style.color = "#ff6b6b";
    } finally {
        btn.disabled = false;
    }
}
