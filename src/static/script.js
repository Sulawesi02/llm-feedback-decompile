function addTestCaseRow() {
    const container = document.getElementById('test-cases-container');
    if (!container) return;
    const row = document.createElement('div');
    row.className = 'test-case-row';

    const input = document.createElement('input');
    input.className = 'test-input';
    input.placeholder = '例如: (int[]){1, 2, 3}, 3';

    const arrow = document.createElement('span');
    arrow.textContent = '→';

    const output = document.createElement('input');
    output.className = 'test-output';
    output.placeholder = '例如: 6';

    const addBtn = document.createElement('button');
    addBtn.type = 'button';
    addBtn.className = 'add-test-case-btn';
    addBtn.textContent = '+';
    addBtn.addEventListener('click', () => {
        addTestCaseRow();
        updateAddButtons();
    });

    const removeBtn = document.createElement('button');
    removeBtn.type = 'button';
    removeBtn.className = 'remove-test-case-btn';
    removeBtn.textContent = '-';
    removeBtn.addEventListener('click', () => {
        row.remove();
        updateAddButtons();
    });

    row.appendChild(input);
    row.appendChild(arrow);
    row.appendChild(output);
    row.appendChild(removeBtn);
    row.appendChild(addBtn);
    container.appendChild(row);
}

function updateAddButtons() {
    const rows = document.querySelectorAll('.test-case-row');
    rows.forEach((row, index) => {
        const addBtn = row.querySelector('.add-test-case-btn');
        const removeBtn = row.querySelector('.remove-test-case-btn');
        
        if (index === rows.length - 1) {
            if (addBtn) addBtn.style.display = '';
            if (removeBtn) removeBtn.style.display = 'none';
        } else {
            if (addBtn) addBtn.style.display = 'none';
            if (removeBtn) removeBtn.style.display = '';
        }
    });
}

function collectTestCases() {
    const rows = document.querySelectorAll('.test-case-row');
    const cases = [];
    rows.forEach(row => {
        const inputEl = row.querySelector('.test-input');
        const outputEl = row.querySelector('.test-output');
        if (!inputEl || !outputEl) return;
        const inputVal = inputEl.value.trim();
        const outputVal = outputEl.value.trim();
        if (!inputVal && !outputVal) return;
        cases.push({ input: inputVal, output: outputVal });
    });
    return cases;
}

document.addEventListener('DOMContentLoaded', () => {
    addTestCaseRow();
    updateAddButtons();
});

async function decompile() {
    const btn = document.getElementById('decompile-btn');
    const output = document.getElementById('output-code');
    
    output.textContent = "// 反编译中...";
    output.style.color = "#f8f8f2"; // Reset color
    btn.disabled = true;

    const arch = document.getElementById('arch').value;
    const machineCode = document.getElementById('machine-code').value.trim();
    const testCases = collectTestCases();

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
                machine_code: machineCode,
                test_cases: testCases
            })
        });

        const data = await response.json();

        if (response.ok) {
            const isSuccess = data.success;
            output.textContent = isSuccess ? data.best_c_code || "// 未生成有效 C 函数代码." : `${data.error || '未知错误'}`;
            output.style.color = isSuccess ? "#f8f8f2" : "#ff6b6b";
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
