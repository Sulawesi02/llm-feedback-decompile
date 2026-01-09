async function decompile() {
    const btn = document.getElementById('decompile-btn');
    const status = document.getElementById('status-msg');
    const output = document.getElementById('output-code');
    
    output.textContent = "// Waiting for server response...";
    output.style.color = "#f8f8f2"; // Reset color
    btn.disabled = true;
    status.style.display = 'block';
    status.textContent = "Processing...";
    status.className = "loading";

    const arch = document.getElementById('arch').value;
    const opt = document.getElementById('opt').value;
    const machineCode = document.getElementById('machine-code').value.trim();

    if (!machineCode) {
        output.textContent = "// Error: Please enter machine code.";
        output.style.color = "#ff6b6b";
        btn.disabled = false;
        status.style.display = 'none';
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
                opt: opt,
                machine_code: machineCode
            })
        });

        const data = await response.json();

        if (response.ok) {
            output.textContent = data.c_code;
        } else {
            output.textContent = `// 服务器错误:\n${data.detail || data.error || '未知错误'}`;
            output.style.color = "#ff6b6b";
        }
    } catch (error) {
        output.textContent = `// 网络错误:\n${error.message}`;
        output.style.color = "#ff6b6b";
    } finally {
        btn.disabled = false;
        status.style.display = 'none';
    }
}
