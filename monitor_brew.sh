#!/bin/bash
# Live brew installation progress monitor

echo "=== Brew Installation Progress Monitor ==="
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    clear
    echo "=== $(date '+%H:%M:%S') ==="
    echo ""
    
    # Check brew processes
    echo "📦 Active Brew Processes:"
    brew_procs=$(ps aux | grep -i "brew\|ruby.*Homebrew" | grep -v grep | wc -l | tr -d ' ')
    echo "   Count: $brew_procs"
    echo ""
    
    # Check compilation processes
    echo "🔨 Active Compilation Processes:"
    clang_procs=$(ps aux | grep -i "clang.*\.c\|make" | grep -v grep | wc -l | tr -d ' ')
    echo "   Count: $clang_procs"
    echo ""
    
    # Check OpenSSL build directory
    echo "📂 OpenSSL Build Directory:"
    if [ -d /private/tmp/opensslA3-* 2>/dev/null ]; then
        openssl_size=$(du -sh /private/tmp/opensslA3-* 2>/dev/null | tail -1 | awk '{print $1}')
        openssl_dir=$(ls -d /private/tmp/opensslA3-* 2>/dev/null | tail -1)
        echo "   Size: $openssl_size"
        echo "   Path: $openssl_dir"
        
        # Count compiled .o files
        o_files=$(find "$openssl_dir" -name "*.o" 2>/dev/null | wc -l | tr -d ' ')
        echo "   Compiled objects: $o_files"
    else
        echo "   Not found (may have finished)"
    fi
    echo ""
    
    # Check installed packages
    echo "✅ Installed Packages:"
    if [ -d /usr/local/Cellar/openssl@3 ]; then
        openssl_installed=$(du -sh /usr/local/Cellar/openssl@3 2>/dev/null | awk '{print $1}')
        echo "   OpenSSL 3.6.0: ✅ $openssl_installed"
    else
        echo "   OpenSSL 3.6.0: ⏳ Installing..."
    fi
    
    if [ -d /usr/local/Cellar/libomp ]; then
        libomp_installed=$(du -sh /usr/local/Cellar/libomp 2>/dev/null | awk '{print $1}')
        echo "   libomp: ✅ $libomp_installed"
    else
        echo "   libomp: ⏳ Waiting..."
    fi
    echo ""
    
    # Check CPU usage
    echo "💻 CPU Usage:"
    cpu_usage=$(top -l 1 | grep "CPU usage" | awk '{print $3}' | sed 's/%//')
    echo "   Current: ${cpu_usage}%"
    echo ""
    
    echo "Refreshing every 3 seconds..."
    sleep 3
done

