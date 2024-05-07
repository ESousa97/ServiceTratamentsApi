// next.config.js
module.exports = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: process.env.NODE_ENV === 'production' ? 'https://server-less-api-python1.vercel.app/api/:path*' : 'http://localhost:5000/api/:path*',
      },
      {
        source: '/process',
        destination: process.env.NODE_ENV === 'production' ? 'https://server-less-api-python1.vercel.app/process' : 'http://localhost:5000/process',
      }
    ]
  },
};
