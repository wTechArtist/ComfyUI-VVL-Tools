/**
 * VVL Loop Control Nodes - Dynamic Input/Output UI
 * 
 * 为VVL循环节点实现动态输入/输出端口管理，参考Easy Use和DynamicBatchAnything的实现
 */

import { app } from "../../../scripts/app.js";

console.log("[VVL Loop] Loading VVL Loop Control extension...");

app.registerExtension({
    name: "VVL.LoopControlUI",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!nodeData) return;
        
        const nodeName = nodeData.name;
        const isForLoopStart = nodeName === "VVL forLoopStart";
        const isForLoopEnd = nodeName === "VVL forLoopEnd";
        
        if (!isForLoopStart && !isForLoopEnd) return;

        console.log(`[VVL Loop] Registering dynamic UI for: ${nodeName}`);

        // 自动管理输入端口：移除末尾未使用的多余输入，但保持最少数量
        function removeUnusedInputsFromEnd(node, minInputs = 2) {
            if (!node.inputs) return;
            
            // 对于forLoopStart，保留total和parallel，然后是initial_value
            // 对于forLoopEnd，保留flow，然后是initial_value
            const fixedInputCount = isForLoopStart ? 2 : 1;
            
            while (node.inputs.length > fixedInputCount + minInputs) {
                const last = node.inputs[node.inputs.length - 1];
                if (!last || last.link == null) {
                    node.removeInput(node.inputs.length - 1);
                } else {
                    break;
                }
            }
        }

        // 确保末尾有一个空的initial_value输入
        function ensureTrailingEmpty(node) {
            if (!node.inputs) node.inputs = [];
            
            // 获取当前initial_value输入的数量
            let initialValueCount = 0;
            for (let i = 0; i < node.inputs.length; i++) {
                if (node.inputs[i].name.startsWith("initial_value")) {
                    initialValueCount++;
                }
            }
            
            // 检查最后一个initial_value是否已连接
            let lastInitialValue = null;
            for (let i = node.inputs.length - 1; i >= 0; i--) {
                if (node.inputs[i].name.startsWith("initial_value")) {
                    lastInitialValue = node.inputs[i];
                    break;
                }
            }
            
            // 如果最后一个initial_value已连接，或者没有initial_value输入，添加新的
            if (!lastInitialValue || lastInitialValue.link != null) {
                const nextIndex = initialValueCount + 1;
                node.addInput(`initial_value${nextIndex}`, "*");
            }
        }

        // 为forLoopStart特殊处理输出端口
        function updateForLoopStartOutputs(node) {
            if (!isForLoopStart) return;
            
            // 计算initial_value输入数量
            let initialValueCount = 0;
            for (let i = 0; i < node.inputs.length; i++) {
                if (node.inputs[i].name.startsWith("initial_value")) {
                    initialValueCount++;
                }
            }
            
            // 更新输出端口：flow, index, value1, value2, ...
            // 至少需要3个输出：flow + index + value1（即使没有initial_value输入）
            const requiredOutputs = 2 + Math.max(initialValueCount, 1); // flow + index + values (至少1个value)
            
            // 添加缺失的输出
            while (node.outputs.length < requiredOutputs) {
                const outputIndex = node.outputs.length;
                if (outputIndex === 0) {
                    node.addOutput("flow", "FLOW_CONTROL");
                } else if (outputIndex === 1) {
                    node.addOutput("index", "INT");
                } else {
                    node.addOutput(`value${outputIndex - 1}`, "*");
                }
            }
            
            // 移除多余的输出
            while (node.outputs.length > requiredOutputs) {
                node.removeOutput(node.outputs.length - 1);
            }
        }

        // 为forLoopEnd特殊处理输出端口
        function updateForLoopEndOutputs(node) {
            if (!isForLoopEnd) return;
            
            // 计算initial_value输入数量（除了flow）
            let initialValueCount = 0;
            for (let i = 0; i < node.inputs.length; i++) {
                if (node.inputs[i].name.startsWith("initial_value")) {
                    initialValueCount++;
                }
            }
            
            // 更新输出端口：value1, value2, ...
            // 至少需要1个输出：value1（即使没有initial_value输入）
            const requiredOutputs = Math.max(initialValueCount, 1);
            
            // 添加缺失的输出
            while (node.outputs.length < requiredOutputs) {
                const outputIndex = node.outputs.length + 1;
                node.addOutput(`value${outputIndex}`, "*");
            }
            
            // 移除多余的输出
            while (node.outputs.length > requiredOutputs) {
                node.removeOutput(node.outputs.length - 1);
            }
        }

        // 稳定节点状态
        nodeType.prototype.stabilize = function() {
            removeUnusedInputsFromEnd(this, 1);
            ensureTrailingEmpty(this);
            
            if (isForLoopStart) {
                updateForLoopStartOutputs(this);
            } else if (isForLoopEnd) {
                updateForLoopEndOutputs(this);
            }
            
            // 强制重新计算和设置节点大小
            if (this.computeSize && this.setSize) {
                const newSize = this.computeSize();
                this.setSize(newSize);
            }
            
            // 标记需要重绘
            this.setDirtyCanvas?.(true, true);
        };

        // 节点创建时的初始化
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            if (onNodeCreated) {
                onNodeCreated.apply(this, arguments);
            }

            // 设置节点颜色 - 参考Easy Use的样式
            if (isForLoopStart) {
                this.color = "#222233";           
                this.bgcolor = "#264653";         
            } else if (isForLoopEnd) {
                this.color = "#264653";          
                this.bgcolor = "#222233";         
            }

            // 初始化输入端口 - 只创建最小必要的端口
            if (!this.inputs) this.inputs = [];
            
            if (isForLoopStart) {
                // forLoopStart初始化：total, parallel, initial_value1 + 对应的输出
                if (this.inputs.length === 0) {
                    this.addInput("total", "INT");
                    this.addInput("parallel", "BOOLEAN");
                    this.addInput("initial_value1", "*");
                }
                // 确保有默认输出：flow, index, value1
                if (this.outputs.length === 0) {
                    this.addOutput("flow", "FLOW_CONTROL");
                    this.addOutput("index", "INT");
                    this.addOutput("value1", "*");
                }
            } else if (isForLoopEnd) {
                // forLoopEnd初始化：flow, initial_value1 + 对应的输出
                if (this.inputs.length === 0) {
                    this.addInput("flow", "FLOW_CONTROL");
                    this.addInput("initial_value1", "*");
                }
                // 确保有默认输出：value1
                if (this.outputs.length === 0) {
                    this.addOutput("value1", "*");
                }
            }
            
            // 调度稳定化和初始大小设置
            setTimeout(() => {
                this.stabilize?.();
                this.setSize?.(this.computeSize?.());
            }, 10);
            
            console.log(`[VVL Loop] Node created and styled: ${nodeName}`);
        };

        // 连接变化时的处理
        nodeType.prototype.onConnectionsChange = function(type, slotIndex, isConnected, linkInfo, ioSlot) {
            if (typeof this._debounceTimer !== "undefined") clearTimeout(this._debounceTimer);
            this._debounceTimer = setTimeout(() => {
                this.stabilize?.();
                // 连接变化后重新计算节点大小
                this.setSize?.(this.computeSize?.());
            }, 64);
        };

        // 连接链变化时的处理
        nodeType.prototype.onConnectionsChainChange = function() {
            if (typeof this._debounceTimer !== "undefined") clearTimeout(this._debounceTimer);
            this._debounceTimer = setTimeout(() => {
                this.stabilize?.();
                // 连接变化后重新计算节点大小
                this.setSize?.(this.computeSize?.());
            }, 64);
        };

        // 序列化时处理 - 确保输入/输出状态正确保存
        const onSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function(o) {
            if (onSerialize) {
                onSerialize.apply(this, arguments);
            }
            
            // 保存initial_value输入的数量，用于反序列化时恢复
            let initialValueCount = 0;
            for (let i = 0; i < this.inputs.length; i++) {
                if (this.inputs[i].name.startsWith("initial_value")) {
                    initialValueCount++;
                }
            }
            
            o.vvl_initial_value_count = initialValueCount;
        };

        // 反序列化时处理 - 恢复正确的输入/输出数量
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(o) {
            if (onConfigure) {
                onConfigure.apply(this, arguments);
            }
            
            // 恢复initial_value输入数量
            if (o.vvl_initial_value_count) {
                const targetCount = o.vvl_initial_value_count;
                let currentCount = 0;
                
                // 计算当前initial_value数量
                for (let i = 0; i < this.inputs.length; i++) {
                    if (this.inputs[i].name.startsWith("initial_value")) {
                        currentCount++;
                    }
                }
                
                // 添加缺失的initial_value输入
                while (currentCount < targetCount) {
                    currentCount++;
                    this.addInput(`initial_value${currentCount}`, "*");
                }
            }
            
            // 调度稳定化和大小调整
            setTimeout(() => {
                this.stabilize?.();
                this.setSize?.(this.computeSize?.());
            }, 10);
        };
    },

    async setup() {
        console.log("[VVL Loop] VVL Loop Control extension loaded!");
    }
});

console.log("[VVL Loop] VVL Loop Control script loaded!");